
import sys

import pytest

from llamafactory.chat import ChatModel
from llamafactory.extras.packages import is_sglang_available


MODEL_NAME = "Qwen/Qwen2.5-0.5B"


INFER_ARGS = {
    "model_name_or_path": MODEL_NAME,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
    "infer_backend": "sglang",
    "do_sample": False,
    "max_new_tokens": 1,
}


MESSAGES = [
    {"role": "user", "content": "Hi"},
]


@pytest.mark.skipif(not is_sglang_available(), reason="SGLang is not installed")
def test_chat():
    r
    chat_model = ChatModel(INFER_ARGS)
    response = chat_model.chat(MESSAGES)[0]
    print(response.response_text)


@pytest.mark.skipif(not is_sglang_available(), reason="SGLang is not installed")
def test_stream_chat():
    r
    chat_model = ChatModel(INFER_ARGS)

    response = ""
    for token in chat_model.stream_chat(MESSAGES):
        response += token

    print("Complete response:", response)
    assert response, "Should receive a non-empty response"


if __name__ == "__main__":
    if not is_sglang_available():
        print("SGLang is not available. Please install it.")
        sys.exit(1)

    test_chat()
    test_stream_chat()
