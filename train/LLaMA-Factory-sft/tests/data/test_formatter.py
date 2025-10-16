
import json
from datetime import datetime

from llamafactory.data.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter


FUNCTION = {"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}

TOOLS = [
    {
        "name": "test_tool",
        "description": "tool_desc",
        "parameters": {
            "type": "object",
            "properties": {
                "foo": {"type": "string", "description": "foo_desc"},
                "bar": {"type": "number", "description": "bar_desc"},
            },
            "required": ["foo"],
        },
    }
]


def test_empty_formatter():
    formatter = EmptyFormatter(slots=["\n"])
    assert formatter.apply() == ["\n"]


def test_string_formatter():
    formatter = StringFormatter(slots=["<s>", "Human: {{content}}\nAssistant:"])
    assert formatter.apply(content="Hi") == ["<s>", "Human: Hi\nAssistant:"]


def test_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}", "</s>"], tool_format="default")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == [
        ,
        "</s>",
    ]


def test_multi_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}", "</s>"], tool_format="default")
    tool_calls = json.dumps([FUNCTION] * 2)
    assert formatter.apply(content=tool_calls) == [
        
        ,
        "</s>",
    ]


def test_default_tool_formatter():
    formatter = ToolFormatter(tool_format="default")
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "You have access to the following tools:\n"
        "> Tool Name: test_tool\n"
        "Tool Description: tool_desc\n"
        "Tool Args:\n"
        "  - foo (string, required): foo_desc\n"
        "  - bar (number): bar_desc\n\n"
        "Use the following format if using a tool:\n"
        "```\n"
        "Action: tool name (one of [test_tool])\n"
        "Action Input: the input to the tool, in a JSON format representing the kwargs "
        
        "```\n"
    ]


def test_default_tool_extractor():
    formatter = ToolFormatter(tool_format="default")
    result = 
    assert formatter.extract(result) == [("test_tool", )]


def test_default_multi_tool_extractor():
    formatter = ToolFormatter(tool_format="default")
    result = (
        
        
    )
    assert formatter.extract(result) == [
        ("test_tool", ),
        ("another_tool", ),
    ]


def test_glm4_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}"], tool_format="glm4")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == []


def test_glm4_tool_formatter():
    formatter = ToolFormatter(tool_format="glm4")
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，"
        "你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具\n\n"
        f"## test_tool\n\n{json.dumps(TOOLS[0], indent=4, ensure_ascii=False)}\n"
        "在调用上述函数时，请使用 Json 格式表示调用的参数。"
    ]


def test_glm4_tool_extractor():
    formatter = ToolFormatter(tool_format="glm4")
    result = 
    assert formatter.extract(result) == [("test_tool", )]


def test_llama3_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == [
        
    ]


def test_llama3_multi_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3")
    tool_calls = json.dumps([FUNCTION] * 2)
    assert formatter.apply(content=tool_calls) == [
        
        
        
    ]


def test_llama3_tool_formatter():
    formatter = ToolFormatter(tool_format="llama3")
    date = datetime.now().strftime("%d %b %Y")
    wrapped_tool = {"type": "function", "function": TOOLS[0]}
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        f"Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
        "You have access to the following functions. "
        "To call a function, please respond with JSON for a function call. "
        
        f"Do not use variables.\n\n{json.dumps(wrapped_tool, indent=4, ensure_ascii=False)}\n\n"
    ]


def test_llama3_tool_extractor():
    formatter = ToolFormatter(tool_format="llama3")
    result = 
    assert formatter.extract(result) == [("test_tool", )]


def test_llama3_multi_tool_extractor():
    formatter = ToolFormatter(tool_format="llama3")
    result = (
        
        
    )
    assert formatter.extract(result) == [
        ("test_tool", ),
        ("another_tool", ),
    ]


def test_mistral_function_formatter():
    formatter = FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", "</s>"], tool_format="mistral")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == [
        "[TOOL_CALLS] " ,
        "</s>",
    ]


def test_mistral_multi_function_formatter():
    formatter = FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", "</s>"], tool_format="mistral")
    tool_calls = json.dumps([FUNCTION] * 2)
    assert formatter.apply(content=tool_calls) == [
        "[TOOL_CALLS] "
        
        ,
        "</s>",
    ]


def test_mistral_tool_formatter():
    formatter = ToolFormatter(tool_format="mistral")
    wrapped_tool = {"type": "function", "function": TOOLS[0]}
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "[AVAILABLE_TOOLS] " + json.dumps([wrapped_tool], ensure_ascii=False) + "[/AVAILABLE_TOOLS]"
    ]


def test_mistral_tool_extractor():
    formatter = ToolFormatter(tool_format="mistral")
    result = 
    assert formatter.extract(result) == [("test_tool", )]


def test_mistral_multi_tool_extractor():
    formatter = ToolFormatter(tool_format="mistral")
    result = (
        
        
    )
    assert formatter.extract(result) == [
        ("test_tool", ),
        ("another_tool", ),
    ]


def test_qwen_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == [
        
    ]


def test_qwen_multi_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen")
    tool_calls = json.dumps([FUNCTION] * 2)
    assert formatter.apply(content=tool_calls) == [
        
        
        "<|im_end|>\n"
    ]


def test_qwen_tool_formatter():
    formatter = ToolFormatter(tool_format="qwen")
    wrapped_tool = {"type": "function", "function": TOOLS[0]}
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        f"\n{json.dumps(wrapped_tool, ensure_ascii=False)}"
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
        
        
    ]


def test_qwen_tool_extractor():
    formatter = ToolFormatter(tool_format="qwen")
    result = 
    assert formatter.extract(result) == [("test_tool", )]


def test_qwen_multi_tool_extractor():
    formatter = ToolFormatter(tool_format="qwen")
    result = (
        
        
    )
    assert formatter.extract(result) == [
        ("test_tool", ),
        ("another_tool", ),
    ]
