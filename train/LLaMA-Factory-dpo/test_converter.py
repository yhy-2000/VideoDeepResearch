
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.data.converter import SharegptDatasetConverter, AlpacaDatasetConverter
from llamafactory.data.data_utils import Role

class MockDatasetAttr:
    def __init__(self):
        self.ranking = True
        self.chosen = "chosen"
        self.rejected = "rejected"
        self.role_tag = "role"
        self.content_tag = "content"
        self.system = None
        self.tools = None
        self.images = None
        self.videos = None
        self.audios = None
        self.user_tag = "user"
        self.assistant_tag = "assistant"
        self.observation_tag = "observation"
        self.function_tag = "function"
        self.system_tag = "system"
        self.messages = "messages"
        self.kto_tag = None
        self.history = None
        self.prompt = None
        self.query = None
        self.response = None

class MockDataArgs:
    def __init__(self):
        self.media_dir = ""

def test_new_format():
    print("Testing new list format for chosen/rejected...")
    
    alpaca_example = {
        "chosen": [
            {
                "from": "human",
                "value": "You are a video understanding expert tasked with analyzing video content and answering single-choice questions..."
            },
            {
                "from": "gpt", 
                "value": "<think>To answer the question...</think>\n<temporal_grounding_agent>Locate the segment(s) where a person is using tools in a workshop.</temporal_grounding_agent>\n[Pause]"
            }
        ],
        "rejected": [
            {
                "from": "human",
                "value": "You are a video understanding expert tasked with analyzing video content and answering single-choice questions..."
            },
            {
                "from": "gpt",
                "value": "<think>To answer the question...</think>\n<temporal_grounding_agent>Locate the segment(s) in the video where a person is using tools in a workshop.</temporal_grounding_agent>\n[Pause]"
            }
        ]
    }
    
    sharegpt_example = {
        "messages": [],
        "chosen": [
            {
                "from": "human",
                "value": "You are a video understanding expert tasked with analyzing video content and answering single-choice questions..."
            },
            {
                "from": "gpt", 
                "value": "<think>To answer the question...</think>\n<temporal_grounding_agent>Locate the segment(s) where a person is using tools in a workshop.</temporal_grounding_agent>\n[Pause]"
            }
        ],
        "rejected": [
            {
                "from": "human",
                "value": "You are a video understanding expert tasked with analyzing video content and answering single-choice questions..."
            },
            {
                "from": "gpt",
                "value": "<think>To answer the question...</think>\n<temporal_grounding_agent>Locate the segment(s) in the video where a person is using tools in a workshop.</temporal_grounding_agent>\n[Pause]"
            }
        ]
    }
    
    print("\n=== Testing AlpacaDatasetConverter ===")
    alpaca_converter = AlpacaDatasetConverter(MockDatasetAttr(), MockDataArgs())
    alpaca_result = alpaca_converter(alpaca_example)
    
    print(f"Prompt: {len(alpaca_result['_prompt'])} messages")
    for i, msg in enumerate(alpaca_result['_prompt']):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
    
    print(f"Response: {len(alpaca_result['_response'])} messages")
    for i, msg in enumerate(alpaca_result['_response']):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
    
    print("\n=== Testing SharegptDatasetConverter ===")
    sharegpt_converter = SharegptDatasetConverter(MockDatasetAttr(), MockDataArgs())
    sharegpt_result = sharegpt_converter(sharegpt_example)
    
    print(f"Prompt: {len(sharegpt_result['_prompt'])} messages")
    for i, msg in enumerate(sharegpt_result['_prompt']):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
    
    print(f"Response: {len(sharegpt_result['_response'])} messages")
    for i, msg in enumerate(sharegpt_result['_response']):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")

if __name__ == "__main__":
    test_new_format()
