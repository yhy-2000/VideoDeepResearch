
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    r
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts: torch.Tensor = torch.zeros((bsz, max_num), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)

    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]
    return seqlens


def get_unpad_data(attention_mask: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", int]:
    r
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def configure_packing(model_args: "ModelArguments", is_trainable: bool) -> None:
    if not is_trainable or not model_args.block_diag_attn:
        return

    import transformers.modeling_flash_attention_utils

    transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
    logger.info_rank0("Using block diagonal attention for sequence packing without cross-attention.")
