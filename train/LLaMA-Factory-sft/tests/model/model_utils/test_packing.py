
import pytest
import torch

from llamafactory.model.model_utils.packing import get_seqlens_in_batch, get_unpad_data


@pytest.mark.parametrize(
    "attention_mask,golden_seq_lens",
    [
        (
            [
                [1, 1, 2, 2, 2, 0],
                [1, 2, 2, 3, 3, 3],
            ],
            [2, 3, 1, 2, 3],
        ),
        (
            [[1]],
            [1],
        ),
    ],
)
def test_get_seqlens_in_batch(attention_mask, golden_seq_lens):
    attention_mask_with_indices = torch.tensor(attention_mask)
    seqlens_in_batch = get_seqlens_in_batch(attention_mask_with_indices)
    assert torch.all(seqlens_in_batch == torch.tensor(golden_seq_lens))


@pytest.mark.parametrize(
    "attention_mask,golden_indices,golden_cu_seqlens,golden_max_seqlen",
    [
        (
            [
                [1, 1, 2, 2, 2, 0],
                [1, 2, 2, 3, 3, 3],
            ],
            [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11],
            [0, 2, 5, 6, 8, 11],
            3,
        ),
        (
            [[1]],
            [0],
            [0, 1],
            1,
        ),
    ],
)
def test_get_unpad_data(attention_mask, golden_indices, golden_cu_seqlens, golden_max_seqlen):
    attention_mask_with_indices = torch.tensor(attention_mask)
    indices, cu_seqlens, max_seqlen_in_batch = get_unpad_data(attention_mask_with_indices)
    assert torch.all(indices == torch.tensor(golden_indices))
    assert torch.all(cu_seqlens == torch.tensor(golden_cu_seqlens, dtype=torch.int32))
    assert max_seqlen_in_batch == golden_max_seqlen
