import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from mmgp import offload
from shared.utils import files_locator as fl
from transformers import AutoTokenizer, T5TokenizerFast, Qwen3ForCausalLM

from models.qwen.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from models.qwen.convert_diffusers_qwen_vae import convert_state_dict as convert_vae_state_dict
from .anima_model import AnimaModel

logger = logging.get_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def conv_state_dict(sd):
    """Convert ComfyUI-style state dict keys if needed."""
    out_sd = {}
    for key, tensor in sd.items():
        new_key = key.replace("model.diffusion_model.", "")
        if new_key.startswith("net."):
            new_key = new_key[len("net."):]
        out_sd[new_key] = tensor
    return out_sd


class model_factory:
    """Factory for loading and running Anima models."""

    @staticmethod
    def _pack_latents(latents):
        """Pack latents from [B, C, 1, H, W] to [B, (H//2)*(W//2), C*4] for transformer."""
        batch_size, num_channels_latents, _, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width):
        """Unpack latents from [B, (H//2)*(W//2), C*4] to [B, C, 1, H, W] for VAE."""
        batch_size, num_patches, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents

    @staticmethod
    def _to_image_tensor(image_like, device):
        if image_like is None:
            return None

        if isinstance(image_like, (list, tuple)):
            if len(image_like) == 0:
                return None
            image_like = image_like[0]

        if not torch.is_tensor(image_like):
            return None

        x = image_like
        if x.ndim == 5:  # [B, C, T, H, W]
            x = x[0, :, 0]
        elif x.ndim == 4:
            if x.shape[0] in (1, 3, 4):  # [C, T, H, W]
                x = x[:, 0]
            elif x.shape[1] in (1, 3, 4):  # [B, C, H, W]
                x = x[0]
            else:
                x = x[0]
        elif x.ndim != 3:
            return None

        if x.shape[0] not in (1, 3, 4):
            return None

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 4:
            x = x[:3]

        x = x.to(device=device, dtype=torch.float32)
        if x.min() >= 0.0 and x.max() > 1.0:
            x = x / 255.0
        if x.min() >= 0.0:
            x = x * 2.0 - 1.0
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _to_mask_tensor(mask_like, device):
        if mask_like is None:
            return None

        if isinstance(mask_like, (list, tuple)):
            if len(mask_like) == 0:
                return None
            mask_like = mask_like[0]

        if not torch.is_tensor(mask_like):
            return None

        x = mask_like
        if x.ndim == 5:  # [B, C, T, H, W]
            x = x[0, 0, 0]
        elif x.ndim == 4:
            if x.shape[1] == 1:  # [B, 1, H, W]
                x = x[0, 0]
            elif x.shape[0] == 1:  # [1, T, H, W]
                x = x[0, 0]
            else:
                x = x[0, 0]
        elif x.ndim == 3:
            if x.shape[0] in (1, 3):
                x = x[0]
            else:
                x = x[0]
        elif x.ndim != 2:
            return None

        x = x.to(device=device, dtype=torch.float32)
        if x.min() >= 0.0 and x.max() > 1.0:
            x = x / 255.0
        if x.min() < 0.0:
            x = (x + 1.0) * 0.5
        return torch.where(x >= 0.5, 1.0, 0.0)

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
        if not transformer_filename:
            raise ValueError("[Anima] No transformer filename provided. Please select an Anima model checkpoint.")

        self.base_model_type = base_model_type
        self.model_def = model_def
        self.dtype = dtype

        # --- Load Transformer ---
        def preprocess_sd(state_dict):
            return conv_state_dict(state_dict)

        default_transformer_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{base_model_type}.json")
        if not os.path.isfile(default_transformer_config):
            raise FileNotFoundError(
                f"[Anima] Config file not found: {default_transformer_config}. "
                f"Expected config for model type '{base_model_type}'."
            )

        try:
            with open(default_transformer_config, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"[Anima] Invalid JSON in config {default_transformer_config}: {exc}") from exc
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        kwargs_light = {"writable_tensors": False, "preprocess_sd": preprocess_sd}

        with init_empty_weights():
            transformer = AnimaModel(**config)

        try:
            if source is not None:
                source_path = fl.locate_file(source)
                offload.load_model_data(transformer, source_path, **kwargs_light)
            else:
                offload.load_model_data(transformer, model_filename, **kwargs_light)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[Anima] Transformer weights not found: {exc}. "
                "Please download the Anima model checkpoint."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"[Anima] Failed to load transformer weights: {exc}") from exc

        transformer.to(dtype)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, dtype, None, None)

        # --- Load Text Encoder (Qwen3 0.6B) ---
        def preprocess_text_encoder_sd(state_dict):
            if 'lm_head.weight' not in state_dict and 'model.embed_tokens.weight' in state_dict:
                state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']
            return state_dict

        try:
            te_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "qwen3_06b_config.json")
            text_encoder = offload.fast_load_transformers_model(
                text_encoder_filename,
                writable_tensors=True,
                modelClass=Qwen3ForCausalLM,
                defaultConfigPath=te_config_file,
                preprocess_sd=preprocess_text_encoder_sd,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[Anima] Text encoder file not found: {exc}. "
                "Please download the Qwen3-0.6B text encoder."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"[Anima] Failed to load text encoder: {exc}") from exc

        # --- Load Tokenizers ---
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"[Anima] Failed to load Qwen3-0.6B tokenizer: {exc}. "
                "Ensure you have internet access or a cached copy of Qwen/Qwen3-0.6B tokenizer."
            ) from exc

        # T5 tokenizer for the LLMAdapter embedding (vocab_size=32128)
        try:
            t5_tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl", legacy=False)
        except Exception as exc:
            raise RuntimeError(
                f"[Anima] Failed to load T5-v1.1-XXL tokenizer: {exc}. "
                "Ensure you have internet access or a cached copy."
            ) from exc

        # --- Load VAE (Qwen Image VAE - AutoencoderKLQwenImage) ---
        try:
            vae_filename = fl.locate_file(os.path.join("Anima", "qwen_image_vae.safetensors"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[Anima] VAE file not found: {exc}. "
                "Please download the Qwen Image VAE."
            ) from exc

        try:
            vae_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen", "configs", "qwen_image_layered_vae_config.json")
            vae = offload.fast_load_transformers_model(
                vae_filename,
                writable_tensors=True,
                modelClass=AutoencoderKLQwenImage,
                defaultConfigPath=vae_config_file,
                configKwargs={"input_channels": 3},
                preprocess_sd=convert_vae_state_dict,
            )
            vae.to(VAE_dtype)
        except Exception as exc:
            raise RuntimeError(f"[Anima] Failed to load VAE: {exc}") from exc
        self.vae = vae

        # --- Scheduler ---
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
        )

        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.scheduler = scheduler

        # Build pipe dict for offload management
        self.pipe = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae": self.vae,
        }

    def _encode_prompt(self, prompt, device="cpu", max_sequence_length=512):
        """Encode a text prompt using the Qwen3 text encoder and T5 tokenizer.
        Returns UNPADDED embeddings at actual text length (matching ComfyUI's pipeline)."""
        if isinstance(prompt, str):
            prompt = [prompt]

        try:
            inputs = self.tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )
        except Exception as exc:
            raise RuntimeError(f"[Anima] Tokenization failed: {exc}") from exc

        input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(device=device, dtype=torch.long)

        # Ensure at least 1 token (empty prompts produce 0 tokens with padding=False)
        if input_ids.shape[1] == 0:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.tensor([[pad_id]], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

        try:
            with torch.no_grad():
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
        except Exception as exc:
            raise RuntimeError(f"[Anima] Text encoding failed: {exc}") from exc

        # T5 token IDs for the LLMAdapter embedding (also unpadded)
        try:
            t5_inputs = self.t5_tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )
            t5_ids = t5_inputs["input_ids"].to(device=device, dtype=torch.long)
            # Ensure at least 1 token for empty prompts
            if t5_ids.shape[1] == 0:
                t5_ids = torch.zeros((1, 1), device=device, dtype=torch.long)
        except Exception as exc:
            raise RuntimeError(f"[Anima] T5 tokenization failed: {exc}") from exc

        return hidden_states, t5_ids

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
        shift=1.0,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loras_slists = kwargs.get("loras_slists")
        try:
            shift = float(shift)
        except (TypeError, ValueError):
            shift = 3.0
        if shift <= 0:
            shift = 3.0
        denoising_strength = float(kwargs.get("denoising_strength", 1.0))
        masking_strength = float(kwargs.get("masking_strength", 1.0))
        denoising_strength = max(0.0, min(1.0, denoising_strength))
        masking_strength = max(0.0, min(1.0, masking_strength))

        input_frames = kwargs.get("input_frames")
        image_start = kwargs.get("image_start")
        input_masks = kwargs.get("input_masks")

        if self._interrupt:
            return None

        if not hasattr(self, 'transformer') or self.transformer is None:
            raise RuntimeError("[Anima] Transformer model not loaded. Please load the model first.")
        if not hasattr(self, 'vae') or self.vae is None:
            raise RuntimeError("[Anima] VAE not loaded. Please load the model first.")

        # Set seed
        generator = torch.Generator(device=device)
        try:
            if seed is None or seed < 0:
                generator.seed()
            else:
                generator.manual_seed(int(seed))
        except Exception as exc:
            logger.warning(f"[Anima] Seed setup failed, using random: {exc}")
            generator.seed()

        # Encode prompts (returns UNPADDED embeddings at actual text length)
        prompt_embeds, prompt_ids = self._encode_prompt(
            input_prompt, device=device, max_sequence_length=max_sequence_length
        )

        negative_embeds = None
        do_cfg = guide_scale > 1.0
        if do_cfg:
            neg = n_prompt if n_prompt and len(n_prompt.strip()) > 0 else ""
            negative_embeds, negative_ids = self._encode_prompt(
                neg, device=device, max_sequence_length=max_sequence_length
            )

        if self._interrupt:
            return None

        # Pre-process through adapter + pad to 512 (matching ComfyUI's extra_conds)
        # ComfyUI runs the adapter on UNPADDED text, then pads to 512 AFTER.
        with torch.no_grad():
            prompt_embeds = self.transformer.preprocess_text_embeds(
                prompt_embeds.to(dtype=self.dtype), prompt_ids
            )
            if prompt_embeds.shape[1] < 512:
                prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, 512 - prompt_embeds.shape[1]))

            if negative_embeds is not None:
                negative_embeds = self.transformer.preprocess_text_embeds(
                    negative_embeds.to(dtype=self.dtype), negative_ids
                )
                if negative_embeds.shape[1] < 512:
                    negative_embeds = F.pad(negative_embeds, (0, 0, 0, 512 - negative_embeds.shape[1]))

        if self._interrupt:
            return None

        # Prepare latents in 5D [B, C, 1, H, W] — AnimaModel handles patching internally
        latent_channels = self.vae.config.z_dim if hasattr(self.vae, 'config') else 16
        latent_h = height // 8
        latent_w = width // 8
        latent_shape = (batch_size, latent_channels, 1, latent_h, latent_w)
        # Keep latents in float32 for stable Euler updates; cast to model dtype only for forward.
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)

        source_latents = None
        image_mask_latents = None
        randn = None
        masked_steps = 0

        init_image = self._to_image_tensor(input_frames, device)
        if init_image is None:
            init_image = self._to_image_tensor(image_start, device)

        if init_image is not None:
            init_image = F.interpolate(
                init_image.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(2)

            vae_dtype = next(self.vae.parameters()).dtype
            with torch.no_grad():
                posterior = self.vae.encode(init_image.to(dtype=vae_dtype), return_dict=True).latent_dist
                z = posterior.sample().to(dtype=torch.float32)

            vae_z_dim = self.vae.config.z_dim
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, vae_z_dim, 1, 1, 1)
                .to(z.device, z.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, vae_z_dim, 1, 1, 1)
                .to(z.device, z.dtype)
            )
            source_latents = (z - latents_mean) / latents_std
            source_latents = F.interpolate(
                source_latents.squeeze(2),
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(2)

            if source_latents.shape[0] != batch_size:
                source_latents = source_latents.repeat(batch_size, 1, 1, 1, 1)

            mask_2d = self._to_mask_tensor(input_masks, device)
            if mask_2d is not None:
                image_mask_latents = F.interpolate(
                    mask_2d.unsqueeze(0).unsqueeze(0),
                    size=(latent_h, latent_w),
                    mode="nearest",
                ).unsqueeze(2)
                image_mask_latents = image_mask_latents.repeat(batch_size, 1, 1, 1, 1)

        # Build sigma schedule matching ComfyUI: ModelSamplingDiscreteFlow with shift=3.0, multiplier=1.0
        SHIFT = shift
        MULTIPLIER = 1.0
        num_internal = 1000
        def time_snr_shift(alpha, t):
            if alpha == 1.0:
                return t
            return alpha * t / (1 + (alpha - 1) * t)
        # Build internal sigma table (matching ModelSamplingDiscreteFlow.set_parameters)
        internal_sigmas = torch.tensor([
            time_snr_shift(SHIFT, (i / num_internal) * MULTIPLIER)
            for i in range(1, num_internal + 1)
        ])
        # simple_scheduler: pick evenly spaced from end to start, append 0
        ss = len(internal_sigmas) / sampling_steps
        sigmas_list = [float(internal_sigmas[-(1 + int(x * ss))]) for x in range(sampling_steps)]
        sigmas_list.append(0.0)
        sigmas = torch.tensor(sigmas_list, device=device)
        num_steps = sampling_steps

        if source_latents is not None:
            first_step = int(round(num_steps * (1.0 - denoising_strength), 4)) if denoising_strength < 1.0 else 0
            first_step = min(max(first_step, 0), max(num_steps - 1, 0))
            randn = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)
            if denoising_strength < 1.0 and len(sigmas) > 1:
                sigma_start = sigmas[first_step]
                latents = source_latents * (1.0 - sigma_start) + randn * sigma_start
                sigmas = sigmas[first_step:]
                num_steps = max(len(sigmas) - 1, 0)
            masked_steps = int(np.ceil(num_steps * masking_strength)) if image_mask_latents is not None else 0

        if loras_slists is not None:
            from shared.utils.loras_mutipliers import update_loras_slists

            update_loras_slists(self.transformer, loras_slists, num_steps)

        # --- Sampler selection ---
        solver = sample_solver.lower().strip() if sample_solver else "er_sde"
        if solver not in ("er_sde", "euler", "euler_a", "dpmpp_2m_sde"):
            logger.warning(f"[Anima] Unknown sampler '{sample_solver}', falling back to er_sde")
            solver = "er_sde"

        # Noise sampler (shared by stochastic samplers)
        noise_generator = torch.Generator(device=device)
        if generator is not None and hasattr(generator, 'initial_seed'):
            noise_generator.manual_seed(generator.initial_seed() + 1)
        def sample_noise():
            return torch.randn(latents.shape, dtype=latents.dtype, device=device, generator=noise_generator)

        # er_sde precomputation
        if solver == "er_sde":
            if sigmas[0] >= 1.0:
                sigmas = sigmas.clone()
                sigmas[0] = time_snr_shift(SHIFT, (1.0 - 1e-4))
            half_log_snrs = sigmas.clamp(1e-7, 1.0 - 1e-7).logit().neg()
            er_lambdas = half_log_snrs.neg().exp()
            def noise_scaler(x):
                return x * ((x ** 0.3).exp() + 10.0)
            num_integration_points = 200.0
            point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=device)

        logger.info(f"[Anima] Denoising: {num_steps} steps, CFG={guide_scale}, sampler={solver}, shift={SHIFT}")
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=num_steps)

        # Pre-compute CFG context (same every step)
        if do_cfg and negative_embeds is not None:
            cfg_context = torch.cat([negative_embeds, prompt_embeds])
        else:
            cfg_context = prompt_embeds

        old_denoised = None
        old_denoised_d = None
        h_last = None

        for i in tqdm(range(num_steps), desc="Diffusing", leave=True):
            if self._interrupt:
                return None

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # --- Common: model forward + CFG ---
            latent_model_input = latents.to(self.dtype)
            timestep = sigma.expand(batch_size)

            if do_cfg and negative_embeds is not None:
                latent_model_input = torch.cat([latent_model_input] * 2)
                timestep = torch.cat([timestep] * 2)

            model_output = self.transformer(
                latent_model_input, timestep, cfg_context,
            ).float()

            if do_cfg and negative_embeds is not None:
                v_uncond, v_cond = model_output.chunk(2)
                model_output = v_uncond + guide_scale * (v_cond - v_uncond)

            # CONST velocity prediction: denoised = x - v * sigma
            denoised = latents - model_output * sigma

            # --- Step update (sampler-specific) ---
            if sigma_next == 0:
                latents = denoised

            elif solver == "euler":
                d = (latents - denoised) / sigma
                latents = latents + d * (sigma_next - sigma)

            elif solver == "euler_a":
                # Ancestral step (matching ComfyUI's get_ancestral_step + sample_euler_ancestral)
                sigma_up = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2).sqrt()
                sigma_down = (sigma_next ** 2 - sigma_up ** 2).sqrt()
                d = (latents - denoised) / sigma
                latents = latents + d * (sigma_down - sigma)
                latents = latents + sample_noise() * sigma_up

            elif solver == "dpmpp_2m_sde":
                # DPM++ 2M SDE (matching ComfyUI's sample_dpmpp_2m_sde)
                t, s = -sigma.log(), -sigma_next.log()
                h = s - t
                eta_h = h  # eta=1.0
                latents = (sigma_next / sigma) * (-eta_h).exp() * latents + (-h - eta_h).expm1().neg() * denoised
                if old_denoised is not None and h_last is not None:
                    r = h_last / h
                    latents = latents + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                # SDE noise
                latents = latents + sample_noise() * sigma_next * (-2 * eta_h).expm1().neg().sqrt()
                h_last = h

            elif solver == "er_sde":
                # Extended Reverse-Time SDE (3-stage)
                stage_used = min(3, i + 1)
                er_lambda_s = er_lambdas[i]
                er_lambda_t = er_lambdas[i + 1]
                alpha_s = 1.0 - sigma.item()
                alpha_t = 1.0 - sigma_next.item()
                r_alpha = alpha_t / alpha_s
                r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

                latents = r_alpha * r * latents + alpha_t * (1 - r) * denoised

                if stage_used >= 2 and old_denoised is not None:
                    dt = er_lambda_t - er_lambda_s
                    lambda_step_size = -dt / num_integration_points
                    lambda_pos = er_lambda_t + point_indice * lambda_step_size
                    scaled_pos = noise_scaler(lambda_pos)
                    s = torch.sum(1 / scaled_pos) * lambda_step_size
                    denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                    latents = latents + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d
                    if stage_used >= 3 and old_denoised_d is not None:
                        s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                        denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                        latents = latents + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                    old_denoised_d = denoised_d

                noise_amt = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
                latents = latents + alpha_t * sample_noise() * noise_amt

            old_denoised = denoised

            if image_mask_latents is not None and source_latents is not None and i < masked_steps:
                noisy_source = source_latents if sigma_next <= 0 else randn * sigma_next + (1.0 - sigma_next) * source_latents
                latents = noisy_source * (1.0 - image_mask_latents) + image_mask_latents * latents

            if callback is not None:
                callback(i, latents[0, :, :1], False)

        if self._interrupt:
            return None

        # Decode latents using Qwen Image VAE (AutoencoderKLQwenImage)
        try:
            # Denormalize latents: latents * std + mean (reverse of encoding normalization)
            vae_z_dim = self.vae.config.z_dim
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, vae_z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, vae_z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            decode_latents = latents * latents_std + latents_mean

            # Cast to VAE dtype
            vae_dtype = next(self.vae.parameters()).dtype
            decode_latents = decode_latents.to(dtype=vae_dtype)
            # Latents are [B, C, 1, H, W] - the 3D VAE expects this format
            images = self.vae.decode(decode_latents, return_dict=False)[0]
        except Exception as exc:
            raise RuntimeError(f"[Anima] VAE decode failed: {exc}") from exc

        # images: [B, C, T, H, W], already clamped by VAE
        # Return first batch item as [C, T, H, W] (T=1 for images)
        result = images[0]  # [C, T, H, W]

        return {"x": result}

    def get_loras_transformer(self, *args, **kwargs):
        get_model_recursive_prop = kwargs.get("get_model_recursive_prop")
        model_type = kwargs.get("model_type")
        model_mode = kwargs.get("model_mode")
        image_mode = kwargs.get("image_mode")

        if get_model_recursive_prop is None:
            return [], []
        if image_mode != 2:
            return [], []

        model_mode_int = None
        if model_mode is not None:
            try:
                model_mode_int = int(model_mode)
            except (TypeError, ValueError):
                model_mode_int = None
        if model_mode_int != 1:
            return [], []

        preloadURLs = get_model_recursive_prop(model_type, "preload_URLs")
        if len(preloadURLs) == 0:
            return [], []
        return [fl.locate_file(os.path.basename(preloadURLs[0]))], [1]

    @property
    def _interrupt(self):
        return getattr(self, "_interrupt_flag", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._interrupt_flag = value
