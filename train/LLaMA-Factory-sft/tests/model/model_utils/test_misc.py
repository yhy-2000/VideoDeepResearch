
import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llamafactory.model.model_utils.misc import find_expanded_modules


HF_TOKEN = os.getenv("HF_TOKEN")


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_expanded_modules():
    config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    expanded_modules = find_expanded_modules(model, ["q_proj", "v_proj"], num_layer_trainable=4)
    assert expanded_modules == [
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.v_proj",
        "model.layers.15.self_attn.q_proj",
        "model.layers.15.self_attn.v_proj",
        "model.layers.23.self_attn.q_proj",
        "model.layers.23.self_attn.v_proj",
        "model.layers.31.self_attn.q_proj",
        "model.layers.31.self_attn.v_proj",
    ]
