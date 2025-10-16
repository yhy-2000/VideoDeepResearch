
import os

from llamafactory.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "template": "llama3",
    "dataset": TINY_DATA,
    "dataset_dir": "ONLINE",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


def test_load_train_only():
    dataset_module = load_dataset_module(**TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is None


def test_load_val_size():
    dataset_module = load_dataset_module(val_size=0.1, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


def test_load_eval_data():
    dataset_module = load_dataset_module(eval_dataset=TINY_DATA, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None
