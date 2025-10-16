
import os

import pytest
import torch

from llamafactory.extras.misc import get_current_device
from llamafactory.train.test_utils import load_train_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.parametrize("disable_gradient_checkpointing", [False, True])
def test_vanilla_checkpointing(disable_gradient_checkpointing: bool):
    model = load_train_model(disable_gradient_checkpointing=disable_gradient_checkpointing, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert getattr(module, "gradient_checkpointing") != disable_gradient_checkpointing


def test_unsloth_gradient_checkpointing():
    model = load_train_model(use_unsloth_gc=True, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert module._gradient_checkpointing_func.__self__.__name__ == "UnslothGradientCheckpointing"


def test_upcast_layernorm():
    model = load_train_model(upcast_layernorm=True, **TRAIN_ARGS)
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name:
            assert param.dtype == torch.float32


def test_upcast_lmhead_output():
    model = load_train_model(upcast_lmhead_output=True, **TRAIN_ARGS)
    inputs = torch.randn((1, 16), dtype=torch.float16, device=get_current_device())
    outputs: torch.Tensor = model.get_output_embeddings()(inputs)
    assert outputs.dtype == torch.float32
