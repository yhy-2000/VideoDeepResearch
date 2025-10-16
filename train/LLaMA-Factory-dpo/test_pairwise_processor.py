
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llamafactory.data.processor.pairwise import PairwiseDatasetProcessor
from src.llamafactory.data.template import get_template_and_fix_tokenizer
from src.llamafactory.hparams import DataArguments


class MockTokenizer:
    def __init__(self):
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.bos_token = "<s>"
        self.bos_token_id = 1
        
    def encode(self, text, add_special_tokens=True):
        words = text.split()
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        ids.extend([hash(word) % 1000 + 10 for word in words])
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids, skip_special_tokens=False):
        text_parts = []
        for id in ids:
            if id == self.bos_token_id and not skip_special_tokens:
                text_parts.append(self.bos_token)
            elif id == self.eos_token_id and not skip_special_tokens:
                text_parts.append(self.eos_token)
            elif id == self.pad_token_id and not skip_special_tokens:
                text_parts.append(self.pad_token)
            elif id > 9:
                text_parts.append(f"token_{id}")
        return " ".join(text_parts)


class MockTemplate:
    def __init__(self):
        self.efficient_eos = True
        self.mm_plugin = MockMMPlugin()
    
    def encode_multiturn(self, tokenizer, messages, system=None, tools=None):
        result = []
        for i in range(0, len(messages), 2):
            if i < len(messages):
                prompt_msg = messages[i]
                prompt_ids = tokenizer.encode(prompt_msg.get("content", ""), add_special_tokens=False)
            else:
                prompt_ids = []
                
            if i + 1 < len(messages):
                response_msg = messages[i + 1]
                response_ids = tokenizer.encode(response_msg.get("content", ""), add_special_tokens=False)
            else:
                response_ids = []
                
            if prompt_ids or response_ids:
                result.append((prompt_ids, response_ids))
        return result
    
    def encode_oneturn(self, tokenizer, messages, system=None, tools=None):
        if len(messages) >= 2:
            prompt_content = " ".join([msg.get("content", "") for msg in messages[:-1]])
            response_content = messages[-1].get("content", "")
        else:
            prompt_content = ""
            response_content = messages[0].get("content", "") if messages else ""
        
        prompt_ids = tokenizer.encode(prompt_content, add_special_tokens=False)
        response_ids = tokenizer.encode(response_content, add_special_tokens=False)
        return prompt_ids, response_ids


class MockMMPlugin:
    def process_messages(self, messages, images, videos, audios, processor):
        return messages
    
    def process_token_ids(self, input_ids, labels, images, videos, audios, tokenizer, processor):
        return input_ids, labels


class MockDataArgs:
    def __init__(self):
        self.cutoff_len = 2048
        self.train_on_prompt = False
        self.mask_history = False


def test_pairwise_processor():
    print("Testing PairwiseDatasetProcessor with multiturn conversation...")
    
    tokenizer = MockTokenizer()
    template = MockTemplate()
    data_args = MockDataArgs()
    
    processor = PairwiseDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=None,
        data_args=data_args
    )
    
    examples = {
        "_prompt": [[
            {"role": "user", "content": "What are the main tools used in a workshop?"},
            {"role": "assistant", "content": "Common workshop tools include hammers, saws, drills, and sanders."},
            {"role": "user", "content": "How many main tools are used by the person in the workshop?"}
        ]],
        "_response": [[
            {"role": "assistant", "content": "Looking at the workshop segments, I can identify 4 main tools: cutting/shaping tool, industrial machine, hammer, and sander."},
            {"role": "assistant", "content": "I can see 3 main tools being used in the workshop."}
        ]],
        "_system": [None],
        "_tools": [None],
        "_images": [[]],
        "_videos": [[]],
        "_audios": [[]]
    }
    
    print("Input examples:")
    print(f"Prompt: {examples['_prompt'][0]}")
    print(f"Chosen response: {examples['_response'][0][0]}")
    print(f"Rejected response: {examples['_response'][0][1]}")
    print()
    
    try:
        result = processor.preprocess_dataset(examples)
        
        print("Processing successful!")
        print(f"Number of processed examples: {len(result['chosen_input_ids'])}")
        
        if len(result['chosen_input_ids']) > 0:
            print("\nFirst example:")
            print(f"Chosen input length: {len(result['chosen_input_ids'][0])}")
            print(f"Rejected input length: {len(result['rejected_input_ids'][0])}")
            print(f"Chosen labels length: {len(result['chosen_labels'][0])}")
            print(f"Rejected labels length: {len(result['rejected_labels'][0])}")
            
            chosen_text = tokenizer.decode(result['chosen_input_ids'][0][:100], skip_special_tokens=False)
            rejected_text = tokenizer.decode(result['rejected_input_ids'][0][:100], skip_special_tokens=False)
            
            print(f"\nChosen text sample: {chosen_text}")
            print(f"Rejected text sample: {rejected_text}")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pairwise_processor()
