


import os
import shutil

import fire
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


def merge_lora(
    base_model_path: str,
    lora_checkpoint_path: str,
    extra_file: str = "spk_dict.pt",
    submodule_name: str = "thinker",
    save_path: str = "./merged_model_checkpoint",
):
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    print("Successfully loaded the original model.")

    if not hasattr(model, submodule_name):
        raise AttributeError(f"The model does not have a submodule named '{submodule_name}'.")

    base_submodule = getattr(model, submodule_name)
    print(f"Successfully extracted submodule: {submodule_name}.")

    lora_model = PeftModel.from_pretrained(base_submodule, lora_checkpoint_path)
    processor = AutoProcessor.from_pretrained(lora_checkpoint_path)
    print("LoRA weights and processor loaded successfully.")

    merged_submodule = lora_model.merge_and_unload()
    print("LoRA weights merged successfully.")

    setattr(model, submodule_name, merged_submodule)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Merged model and tokenizer saved to {save_path}.")

    source_file = os.path.join(base_model_path, extra_file)
    target_file = os.path.join(save_path, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"File '{extra_file}' copied from {base_model_path} to {save_path}.")
    else:
        print(f"File '{extra_file}' not found in {base_model_path}, skipping copy.")


def save_full_model(
    saved_thinker_path: str,
    base_model_path: str,
    save_path: str = "./merged_model_checkpoint",
    extra_file: str = "spk_dict.pt",
):
    
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        saved_thinker_path, torch_dtype="auto", device_map="cpu"
    )
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype="auto", device_map="cpu"
    )
    base_model.thinker = thinker

    processor = AutoProcessor.from_pretrained(saved_thinker_path)
    base_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Merged model and processor saved to {save_path}.")

    source_file = os.path.join(base_model_path, extra_file)
    target_file = os.path.join(save_path, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"File '{extra_file}' copied from {base_model_path} to {save_path}.")
    else:
        print(f"File '{extra_file}' not found in {base_model_path}, skipping copy.")


if __name__ == "__main__":
    fire.Fire({"save_full": save_full_model, "merge_lora": merge_lora})
