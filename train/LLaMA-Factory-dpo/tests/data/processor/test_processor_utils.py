

import pytest

from llamafactory.data.processor.processor_utils import infer_seqlen


@pytest.mark.parametrize(
    "test_input,test_output",
    [
        ((3000, 2000, 1000), (600, 400)),
        ((2000, 3000, 1000), (400, 600)),
        ((1000, 100, 1000), (900, 100)),
        ((100, 1000, 1000), (100, 900)),
        ((100, 500, 1000), (100, 500)),
        ((500, 100, 1000), (500, 100)),
        ((10, 10, 1000), (10, 10)),
    ],
)
def test_infer_seqlen(test_input: tuple[int, int, int], test_output: tuple[int, int]):
    assert test_output == infer_seqlen(*test_input)
