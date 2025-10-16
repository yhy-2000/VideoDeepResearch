
from transformers import Llama4Config, Llama4ForConditionalGeneration, Llama4TextConfig, Llama4VisionConfig


if __name__ == "__main__":
    vision_config = Llama4VisionConfig(
        hidden_size=1408,
        image_size=336,
        intermediate_size=5632,
        num_attention_heads=16,
        num_hidden_layers=4,
        vision_output_dim=4096,
    )
    text_config = Llama4TextConfig(
        hidden_size=512,
        intermediate_size=1024,
        intermediate_size_mlp=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=512 // 8,
        num_local_experts=2,
    )
    config = Llama4Config(vision_config=vision_config, text_config=text_config)
    model = Llama4ForConditionalGeneration._from_config(config)
    model.save_pretrained("tiny-llama4")
