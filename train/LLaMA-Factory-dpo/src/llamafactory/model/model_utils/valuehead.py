
from typing import TYPE_CHECKING

import torch
from transformers.utils import cached_file

from ...extras import logging
from ...extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def load_valuehead_params(path_or_repo_id: str, model_args: "ModelArguments") -> dict[str, torch.Tensor]:
    r
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}
    err_text = ""

    try:
        from safetensors import safe_open

        vhead_file = cached_file(filename=V_HEAD_SAFE_WEIGHTS_NAME, **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = cached_file(filename=V_HEAD_WEIGHTS_NAME, **kwargs)
        return torch.load(vhead_file, map_location="cpu", weights_only=True)
    except Exception as err:
        err_text = str(err)

    logger.info_rank0(f"Provided path ({path_or_repo_id}) does not contain value head weights: {err_text}.")
    logger.info_rank0("Ignore the above message if you are not resuming the training of a value head model.")
    return None


def prepare_valuehead_model(model: "PreTrainedModel") -> None:
    if getattr(model.config, "model_type", None) == "llava":
        setattr(model, "lm_head", model.language_model.get_output_embeddings())
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if getattr(model.config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if getattr(model.config, "model_type", None) == "internlm2":
        setattr(model, "lm_head", model.output)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])
