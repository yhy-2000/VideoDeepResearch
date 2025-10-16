
from typing import TYPE_CHECKING

from ...extras import logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


def configure_kv_cache(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable:
        setattr(config, "use_cache", model_args.use_cache)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", model_args.use_cache)

        if model_args.use_cache:
            logger.info_rank0("KV cache is enabled for faster generation.")
        else:
            logger.info_rank0("KV cache is disabled.")
    else:
        setattr(config, "use_cache", False)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", False)

        logger.info_rank0("KV cache is disabled during training.")
