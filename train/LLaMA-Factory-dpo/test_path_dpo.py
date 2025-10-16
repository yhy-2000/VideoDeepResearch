

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.data.converter import PathDPODatasetConverter
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

def test_path_dpo_converter():
    print("Testing PathDPODatasetConverter for multi-turn conversations...")
    
    test_example = {
        "chosen": [
            {
                "from": "human",
                "value": "Hello, I need help with a complex math problem involving calculus."
            },
            {
                "from": "gpt", 
                "value": "I'd be happy to help you with your calculus problem! What specific topic or problem are you working on?"
            },
            {
                "from": "human",
                "value": "I need to find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"
            },
            {
                "from": "gpt",
                "value": "To find the derivative of f(x) = x³ + 2x² - 5x + 3, I'll use the power rule.\n\nf'(x) = 3x² + 4x - 5\n\nThe derivative is f'(x) = 3x² + 4x - 5."
            }
        ],
        "rejected": [
            {
                "from": "human",
                "value": "Hello, I need help with a complex math problem involving calculus."
            },
            {
                "from": "gpt", 
                "value": "I'd be happy to help you with your calculus problem! What specific topic or problem are you working on?"
            },
            {
                "from": "human",
                "value": "I need to find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"
            },
            {
                "from": "gpt",
                "value": "The derivative is f'(x) = 3x² + 4x - 5. That's it."
            }
        ]
    }
    
    converter = PathDPODatasetConverter(MockDatasetAttr(), MockDataArgs())
    result = converter(test_example)
    
    print("=== Converter Output ===")
    print(f"Chosen path: {len(result['_chosen_path'])} messages")
    for i, msg in enumerate(result['_chosen_path']):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
    
    print(f"\nRejected path: {len(result['_rejected_path'])} messages")
    for i, msg in enumerate(result['_rejected_path']):
        print(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
    
    print(f"\nSystem: {result['_system']}")
    print(f"Tools: {result['_tools']}")
    
    assert len(result['_chosen_path']) == 4, f"Expected 4 messages in chosen path, got {len(result['_chosen_path'])}"
    assert len(result['_rejected_path']) == 4, f"Expected 4 messages in rejected path, got {len(result['_rejected_path'])}"
    
    assert result['_chosen_path'][0]['role'] == Role.USER.value
    assert result['_chosen_path'][1]['role'] == Role.ASSISTANT.value
    assert result['_chosen_path'][2]['role'] == Role.USER.value
    assert result['_chosen_path'][3]['role'] == Role.ASSISTANT.value
    
    assert result['_rejected_path'][0]['role'] == Role.USER.value
    assert result['_rejected_path'][1]['role'] == Role.ASSISTANT.value
    assert result['_rejected_path'][2]['role'] == Role.USER.value
    assert result['_rejected_path'][3]['role'] == Role.ASSISTANT.value
    
    print("\n✅ PathDPODatasetConverter test passed!")
    
if __name__ == "__main__":
    test_path_dpo_converter()
    print("\n🎉 All tests passed! Path-based DPO multi-turn conversation support is ready.")
