
import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.packages import is_requests_available


if is_requests_available():
    import requests


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead


def get_rewards_from_server(server_url: str, messages: list[str]) -> list["torch.Tensor"]:
    r
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    r
    v_head_layer = model.v_head.summary
    if is_deepspeed_zero3_enabled():
        import deepspeed

        params = [v_head_layer.weight, v_head_layer.bias]
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    model.pretrained_model.set_adapter(target)
    with context_maybe_zero3:
        if target == "reward":
            setattr(model, "default_head_weight", v_head_layer.weight.data.detach().clone())
            setattr(model, "default_head_bias", v_head_layer.bias.data.detach().clone())

        device = v_head_layer.weight.device
        v_head_layer.weight.data = model.get_buffer(f"{target}_head_weight").detach().clone().to(device)
        v_head_layer.bias.data = model.get_buffer(f"{target}_head_bias").detach().clone().to(device)


def dump_layernorm(model: "PreTrainedModel") -> dict[str, "torch.Tensor"]:
    r
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)

    return layer_norm_params


def restore_layernorm(model: "PreTrainedModel", layernorm_params: Optional[dict[str, "torch.Tensor"]] = None) -> None:
    r
    for name, param in model.named_parameters():
        if name in layernorm_params:
            param.data = layernorm_params[name]
