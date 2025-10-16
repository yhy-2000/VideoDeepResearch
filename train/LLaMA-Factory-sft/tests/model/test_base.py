
import os

import pytest

from llamafactory.train.test_utils import compare_model, load_infer_model, load_reference_model, patch_valuehead_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_VALUEHEAD = os.getenv("TINY_LLAMA_VALUEHEAD", "llamafactory/tiny-random-Llama-3-valuehead")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "template": "llama3",
    "infer_dtype": "float16",
}


@pytest.fixture
def fix_valuehead_cpu_loading():
    patch_valuehead_model()


def test_base():
    model = load_infer_model(**INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA3)
    compare_model(model, ref_model)


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_valuehead():
    model = load_infer_model(add_valuehead=True, **INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_VALUEHEAD, add_valuehead=True)
    compare_model(model, ref_model)
