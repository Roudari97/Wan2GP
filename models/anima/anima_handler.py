import os
import shutil
import tempfile

import torch
from huggingface_hub import hf_hub_download
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "image_outputs": True,
            "guidance_max_phases": 1,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [],
            "sample_solvers": [
                ("er_sde (sharp, flat colors)", "er_sde"),
                ("euler_a (soft, thin lines)", "euler_a"),
                ("dpmpp_2m_sde (creative)", "dpmpp_2m_sde"),
                ("euler", "euler"),
            ],
        }

        extra_model_def["source"] = os.path.join("Anima", "anima-preview.safetensors")
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("circlestone-labs/Anima", "split_files/text_encoders", "qwen_3_06b_base.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = "Anima"

        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["anima_preview"]

    @staticmethod
    def query_family_maps():
        models_eqv_map = {}
        models_comp_map = {}
        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "anima"

    @staticmethod
    def query_family_infos():
        return {"anima": (130, "Anima")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-anima",
            type=str,
            default=None,
            help=f"Path to a directory that contains Anima LoRAs (default: {os.path.join(lora_root, 'anima')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_anima", None) or os.path.join(lora_root, "anima")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        target_dir = os.path.join(fl.get_download_location(), "Anima")
        repo_id = "circlestone-labs/Anima"
        files = [
            ("split_files/diffusion_models", "anima-preview.safetensors"),
            ("split_files/text_encoders", "qwen_3_06b_base.safetensors"),
            ("split_files/vae", "qwen_image_vae.safetensors"),
        ]
        for subfolder, filename in files:
            local_path = os.path.join(target_dir, filename)
            if not os.path.isfile(local_path):
                os.makedirs(target_dir, exist_ok=True)
                with tempfile.TemporaryDirectory() as tmp:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder,
                        local_dir=tmp,
                    )
                    downloaded = os.path.join(tmp, *subfolder.split("/"), filename)
                    shutil.move(downloaded, local_path)
        return []

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        from .anima_main import model_factory

        if not model_filename:
            raise ValueError("[Anima] No model filename provided. Please select an Anima model checkpoint.")
        if not text_encoder_filename:
            raise ValueError("[Anima] No text encoder filename provided. Ensure Qwen3-0.6B is downloaded.")

        try:
            pipe_processor = model_factory(
                checkpoint_dir="ckpts",
                model_filename=model_filename,
                model_type=model_type,
                model_def=model_def,
                base_model_type=base_model_type,
                text_encoder_filename=text_encoder_filename,
                quantizeTransformer=quantizeTransformer,
                dtype=dtype,
                VAE_dtype=VAE_dtype,
                mixed_precision_transformer=mixed_precision_transformer,
                save_quantized=save_quantized,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[Anima] Required model file not found: {exc}. "
                "Please ensure all Anima model files are downloaded via the Downloads tab."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"[Anima] Failed to load model: {exc}") from exc

        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.vae,
        }
        return pipe_processor, pipe

    @staticmethod
    def get_rgb_factors(base_model_type):
        try:
            from shared.RGB_factors import get_rgb_factors
            latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("flux")
            return latent_rgb_factors, latent_rgb_factors_bias
        except Exception as exc:
            print(f"[Anima] Warning: failed to load RGB factors, using None: {exc}")
            return None, None

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        pass

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "guidance_scale": 4,
                "num_inference_steps": 30,
                "flow_shift": 1.0,
                "sample_solver": "er_sde",
            }
        )
