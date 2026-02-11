import json
import os
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from mmgp import offload
from shared.utils import files_locator as fl
from transformers import AutoTokenizer, Qwen3ForCausalLM

from .anima_model import AnimaModel

logger = logging.get_logger(__name__)


def conv_state_dict(sd):
    """Convert ComfyUI-style state dict keys if needed."""
    out_sd = {}
    for key, tensor in sd.items():
        new_key = key.replace("model.diffusion_model.", "")
        out_sd[new_key] = tensor
    return out_sd


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        **kwargs,
    ):
        model_def = model_def or {}
        source = model_def.get("source", None)

        transformer_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        if transformer_filename is None:
            raise ValueError("No transformer filename provided for Anima.")

        self.base_model_type = base_model_type
        self.model_def = model_def
        self.dtype = dtype

        # --- Load Transformer ---
        def preprocess_sd(state_dict):
            return conv_state_dict(state_dict)

        # Default config for Anima 2B
        config = {
            "max_img_h": 1024,
            "max_img_w": 1024,
            "max_frames": 1,
            "in_channels": 16,
            "out_channels": 16,
            "patch_spatial": 2,
            "patch_temporal": 1,
            "concat_padding_mask": True,
            "model_channels": 2048,
            "num_blocks": 28,
            "num_heads": 32,
            "mlp_ratio": 4.0,
            "crossattn_emb_channels": 1024,
            "pos_emb_cls": "rope3d",
            "pos_emb_learnable": False,
            "pos_emb_interpolation": "crop",
            "rope_h_extrapolation_ratio": 1.0,
            "rope_w_extrapolation_ratio": 1.0,
            "rope_t_extrapolation_ratio": 1.0,
        }

        # Check for custom config file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{base_model_type}.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                custom_config = json.load(f)
            custom_config.pop("_class_name", None)
            custom_config.pop("_diffusers_version", None)
            config.update(custom_config)

        kwargs_light = {"writable_tensors": False, "preprocess_sd": preprocess_sd}

        with init_empty_weights():
            transformer = AnimaModel(**config)

        if source is not None:
            offload.load_model_data(transformer, fl.locate_file(source), **kwargs_light)
        else:
            offload.load_model_data(transformer, model_filename, **kwargs_light)

        transformer.to(dtype)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, dtype, None, None)

        # --- Load Text Encoder (Qwen3 0.6B) ---
        text_encoder = offload.fast_load_transformers_model(
            text_encoder_filename,
            writable_tensors=True,
            modelClass=Qwen3ForCausalLM,
        )

        # --- Load Tokenizer ---
        text_encoder_folder = model_def.get("text_encoder_folder")
        if text_encoder_folder:
            tokenizer_path = os.path.dirname(fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json")))
        else:
            tokenizer_path = os.path.dirname(text_encoder_filename)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # --- Load VAE (Wan2.1 VAE for Wan21 latent format) ---
        from models.wan.modules.vae import WanVAE
        vae_urls = model_def.get("VAE_URLs", [])
        if isinstance(vae_urls, list) and len(vae_urls) > 0:
            vae_file = fl.locate_file(os.path.basename(vae_urls[0]))
        else:
            vae_file = "Wan2.1_VAE.safetensors"
        self.vae = WanVAE(vae_pth=fl.locate_file(vae_file), dtype=VAE_dtype, device="cpu")
        self.vae.device = "cpu"

        # --- Scheduler ---
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        # Build pipe dict for offload management
        self.pipe = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae": self.vae.model,
        }

    def _encode_prompt(self, prompt, device="cpu", max_sequence_length=512):
        """Encode a text prompt using the Qwen3 text encoder."""
        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Use the last hidden state
            hidden_states = outputs.hidden_states[-1]

        return hidden_states, input_ids

    def generate(
        self,
        seed=None,
        input_prompt="",
        n_prompt="",
        sampling_steps=30,
        sample_solver="default",
        width=1024,
        height=1024,
        guide_scale=4.0,
        batch_size=1,
        callback=None,
        max_sequence_length=512,
        VAE_tile_size=0,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._interrupt:
            return None

        # Set seed
        generator = torch.Generator(device=device)
        if seed is None or seed < 0:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        # Encode prompts
        prompt_embeds, prompt_ids = self._encode_prompt(
            input_prompt, device=device, max_sequence_length=max_sequence_length
        )

        negative_embeds = None
        negative_ids = None
        do_cfg = guide_scale > 1.0
        if do_cfg:
            neg = n_prompt if n_prompt and len(n_prompt.strip()) > 0 else ""
            negative_embeds, negative_ids = self._encode_prompt(
                neg, device=device, max_sequence_length=max_sequence_length
            )

        if self._interrupt:
            return None

        # Prepare latents (Wan21 format: 16 channels, spatial downsample 8x)
        latent_h = height // 8
        latent_w = width // 8
        latent_shape = (batch_size, 16, 1, latent_h, latent_w)
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=self.dtype)

        # Setup scheduler
        self.scheduler.set_timesteps(sampling_steps, device=device)
        timesteps = self.scheduler.timesteps

        callback(-1, None, True, override_num_inference_steps=sampling_steps)

        # Denoising loop
        for i, t in enumerate(timesteps):
            if self._interrupt:
                return None

            latent_model_input = latents
            timestep = t.expand(batch_size)

            # CFG: concatenate unconditional and conditional inputs
            if do_cfg and negative_embeds is not None:
                latent_model_input = torch.cat([latent_model_input] * 2)
                timestep = torch.cat([timestep] * 2)
                context = torch.cat([negative_embeds, prompt_embeds])
                t5xxl_ids = torch.cat([negative_ids, prompt_ids])
            else:
                context = prompt_embeds
                t5xxl_ids = prompt_ids

            # Predict velocity
            velocity = self.transformer(
                latent_model_input,
                timestep,
                context,
                t5xxl_ids=t5xxl_ids,
            )

            # CFG
            if do_cfg and negative_embeds is not None:
                velocity_uncond, velocity_cond = velocity.chunk(2)
                velocity = velocity_uncond + guide_scale * (velocity_cond - velocity_uncond)

            # Scheduler step
            latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]

            if callback is not None:
                # Preview: pass first frame latent for preview
                latents_preview = latents[:, :, :1]
                callback(i, latents_preview[0], False)

        if self._interrupt:
            return None

        # Decode latents using Wan VAE
        # WanVAE.decode handles normalization internally via self.scale
        # Unbind batch dim: list of [C, T, H, W] tensors
        x0 = latents.unbind(dim=0)
        # Take only first frame for image output
        x0 = [x[:, :1] for x in x0]

        videos = self.vae.decode(x0, VAE_tile_size)

        # Return format: [C, T, H, W] matching Wan model image output
        if len(videos) > 1:
            result = torch.cat([video[:, :1] for video in videos], dim=1)
        else:
            result = videos[0][:, :1]

        return {"x": result}

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        return getattr(self, "_interrupt_flag", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._interrupt_flag = value
