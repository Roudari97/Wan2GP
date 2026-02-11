import json
import os
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from mmgp import offload
from shared.utils import files_locator as fl
from transformers import AutoTokenizer, Qwen3ForCausalLM

from models.z_image.autoencoder_kl import AutoencoderKL
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

        # --- Load VAE (Qwen Image VAE - AutoencoderKL) ---
        vae_filename = fl.locate_file("qwen_image_vae.safetensors")
        vae_config_path = fl.locate_file("ZImageTurbo_VAE_bf16_config.json")

        vae = offload.fast_load_transformers_model(
            vae_filename,
            writable_tensors=True,
            modelClass=AutoencoderKL,
            defaultConfigPath=vae_config_path,
            default_dtype=VAE_dtype,
        )
        self.vae = vae

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
            "vae": self.vae,
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

        # Prepare latents
        # AutoencoderKL uses latent_channels from config (typically 16), spatial downsample 8x
        latent_channels = self.vae.config.latent_channels if hasattr(self.vae, 'config') else 16
        latent_h = height // 8
        latent_w = width // 8
        # Transformer expects [B, C, T, H, W] with T=1 for images
        latent_shape = (batch_size, latent_channels, 1, latent_h, latent_w)
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

        # Decode latents using Qwen Image VAE (AutoencoderKL)
        # Squeeze temporal dim: [B, C, 1, H, W] -> [B, C, H, W]
        decode_latents = latents.squeeze(2)

        # Unscale latents
        if hasattr(self.vae, 'config'):
            decode_latents = (decode_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        images = self.vae.decode(decode_latents, return_dict=False)[0]
        # images: [B, C, H, W], clamp to [-1, 1]
        images = images.clamp(-1, 1)

        # Return as [C, T, H, W] for single image (matching Wan2GP image output convention)
        # Add temporal dim back: [B, C, H, W] -> [C, 1, H, W] (first batch item)
        result = images[0].unsqueeze(1)  # [C, 1, H, W]

        return {"x": result}

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        return getattr(self, "_interrupt_flag", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._interrupt_flag = value
