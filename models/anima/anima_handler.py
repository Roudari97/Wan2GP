import os
import torch
from shared.utils.hf import build_hf_url


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "image_outputs": True,
            "guidance_max_phases": 1,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [],
        }

        text_encoder_folder = "Qwen3_06B"
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("circlestone-labs/Anima", "split_files/text_encoders", "qwen_3_06b_base.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = text_encoder_folder

        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["anima"]

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
        download_def = [
            {
                "repoId": "Qwen/Qwen3-0.6B",
                "sourceFolderList": [""],
                "fileList": [
                    ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
                ],
                "targetFolderList": ["Qwen3_06B"],
            }
        ]
        return download_def

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

        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.vae.model,
        }
        return pipe_processor, pipe

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("wan")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        pass

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "guidance_scale": 4,
                "num_inference_steps": 30,
            }
        )
