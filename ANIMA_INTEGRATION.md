# Anima 2B Preview — Integration Notes

## Overview

This document describes the integration of the **Anima 2B Preview** text-to-image model into the Wan2GP project. The work was done on the `feature/anima-2b-preview` branch.

Anima is a diffusion model built on the **Cosmos Predict2 MiniTrainDIT** architecture, using a **Qwen3 0.6B** text encoder and the **Qwen Image VAE** (`AutoencoderKL`) for latent decoding. It was originally released by CircleStone Labs with ComfyUI support; this integration brings it into Wan2GP's model handler framework.

---

## Files Added

### `models/anima/__init__.py`
Empty package init file.

### `models/anima/anima_model.py`
Core transformer implementation (~628 lines). Contains:

- **`MiniTrainDIT`** — The base diffusion transformer from Cosmos Predict2. Implements:
  - 3D patchification (spatial + temporal) of latent inputs
  - Rope3D positional embeddings for spatial/temporal token positions
  - AdaLN (Adaptive Layer Normalization) modulated transformer blocks
  - Timestep embedding via sinusoidal encoding + MLP
  - Self-attention and cross-attention using Wan2GP's `shared.attention.pay_attention` for VRAM-optimized attention dispatch

- **`LLMAdapter`** — Bridges the Qwen3 0.6B text encoder output to the diffusion model's cross-attention input. Uses:
  - Learned token embeddings
  - Rotary positional embeddings
  - Self-attention + cross-attention transformer blocks
  - RMSNorm throughout

- **`AnimaModel`** — Subclass of `MiniTrainDIT` that wires in the `LLMAdapter`. Overrides `forward()` to preprocess text embeddings through the adapter before passing them to the base transformer.

### `models/anima/anima_main.py`
Model factory and generation pipeline. Handles:

- **Transformer loading** — Uses `offload.load_model_data` with `accelerate.init_empty_weights` for memory-efficient loading. Supports ComfyUI-style state dict key conversion (`model.diffusion_model.` prefix stripping).
- **Text encoder loading** — Loads Qwen3 0.6B via `offload.fast_load_transformers_model` with `Qwen3ForCausalLM`.
- **Tokenizer** — Loaded from the `Qwen3_06B/` folder (downloaded from `Qwen/Qwen3-0.6B` on HuggingFace).
- **VAE** — Uses the Qwen Image VAE (`qwen_image_vae.safetensors`) loaded as `AutoencoderKL` from `models.z_image.autoencoder_kl`. This is the same VAE architecture used by Z-Image / Qwen-Image models. Latent unscaling uses the VAE config's `scaling_factor` and `shift_factor`.
- **Scheduler** — `FlowMatchEulerDiscreteScheduler` with shift=3.0.
- **`generate()` method** — Implements the full inference loop:
  - Prompt encoding via Qwen3 with last hidden state extraction
  - Classifier-free guidance (CFG) with configurable scale
  - Latent preview callbacks compatible with Wan2GP's progress UI
  - Returns `{"x": tensor}` dict matching the Wan model output convention

### `models/anima/anima_handler.py`
Family handler that registers Anima with Wan2GP's model system. Implements all required handler methods:

| Method | Purpose |
|--------|---------|
| `query_model_def` | Defines text encoder URLs, folder, `image_outputs: True`, guidance phases |
| `query_supported_types` | Returns `["anima"]` |
| `query_family_maps` | No equivalence/compatibility maps needed |
| `query_model_family` | Returns `"anima"` |
| `query_family_infos` | Display order 130, label "Anima" |
| `query_model_files` | Downloads tokenizer files from `Qwen/Qwen3-0.6B` to `Qwen3_06B/` |
| `load_model` | Delegates to `anima_main.model_factory`, returns `(pipe_processor, pipe)` |
| `get_rgb_factors` | Uses Wan latent RGB factors for preview |
| `set_cache_parameters` | No-op (step-skipping cache not tuned yet) |
| `update_default_settings` | Sets `guidance_scale: 4`, `num_inference_steps: 30` |

### `defaults/anima_preview.json`
Model definition JSON that tells Wan2GP about the Anima model:
- **Name**: "Anima 2B Preview"
- **Architecture**: `"anima"` (maps to the handler)
- **URL**: Points to `circlestone-labs/Anima` on HuggingFace for the diffusion model weights

---

## Files Modified

### `wgp.py`
Added `"models.anima.anima_handler"` to the `family_handlers` list (line 1881). This is the single registration point that makes the model visible to the UI and download/load pipeline.

---

## Design Decisions

### Why Qwen Image VAE?
Anima uses the Qwen Image VAE (`qwen_image_vae.safetensors`), the same `AutoencoderKL` used by Z-Image / Qwen-Image. We import it from `models.z_image.autoencoder_kl` to avoid duplicating the class. The VAE config JSON (`ZImageTurbo_VAE_bf16_config.json`) is downloaded from the Z-Image repo since it defines the same architecture.

### Why not a custom pipeline class?
Z-Image uses a `DiffusionPipeline` subclass (`ZImagePipeline`), but Anima's inference loop is simpler — single-frame image generation with standard CFG. The `generate()` method on `model_factory` is sufficient and avoids unnecessary abstraction.

### Why download tokenizer from Qwen/Qwen3-0.6B?
The `circlestone-labs/Anima` repo only contains the diffusion model weights and text encoder weights. The tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.) are standard Qwen3 files and are sourced from the official Qwen repo.

### Attention dispatch
The model uses Wan2GP's `shared.attention.pay_attention` for self-attention in the transformer blocks. This routes through the project's optimized attention backend (flash-attention, SDPA, etc.) rather than using raw `F.scaled_dot_product_attention`. The `LLMAdapter` uses `F.scaled_dot_product_attention` directly since it runs during text encoding, not the latent denoising loop.

### Output format
The `generate()` method returns `{"x": tensor}` where tensor is `[C, T, H, W]` — matching the Wan model's return convention. This ensures compatibility with Wan2GP's post-processing pipeline (color correction, upsampling, saving).

---

## Model Weights

| Component | Source | Size |
|-----------|--------|------|
| Diffusion model | `circlestone-labs/Anima/split_files/diffusion_models/anima-preview.safetensors` | ~4 GB |
| Text encoder | `circlestone-labs/Anima/split_files/text_encoders/qwen_3_06b_base.safetensors` | ~1.2 GB |
| VAE | `circlestone-labs/Anima/split_files/vae/qwen_image_vae.safetensors` | ~254 MB |
| Tokenizer | `Qwen/Qwen3-0.6B` (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt) | ~11 MB |

---

## Known Limitations / Future Work

- **LLMAdapter config is hardcoded** — The adapter's embedding vocab size (32128), layer count (6), and dimensions are baked into the class. If future Anima variants change these, the config should be externalized to a JSON file.
- **No LoRA support tested** — The handler stubs are in place (`register_lora_cli_args`, `get_lora_dir`, `get_loras_transformer`) but no Anima LoRAs exist yet.
- **Step-skipping cache** — `set_cache_parameters` is a no-op. TeaCache/MagCache coefficients would need to be tuned for Anima's architecture.
- **VAE config dependency** — The VAE config JSON (`ZImageTurbo_VAE_bf16_config.json`) is downloaded from the Z-Image repo. If that repo becomes unavailable, the config would need to be bundled locally.
- **Runtime testing** — The integration compiles and follows all Wan2GP conventions, but end-to-end inference has not been verified yet. Weight loading mismatches (if any) will surface as clear errors on first run.
