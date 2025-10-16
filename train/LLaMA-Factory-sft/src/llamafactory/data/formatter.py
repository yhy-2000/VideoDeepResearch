
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

from typing_extensions import override

from .data_utils import SLOTS
from .tool_utils import FunctionCall, get_tool_utils


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Optional[str] = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        r
        ...

    def extract(self, content: str) -> Union[str, list["FunctionCall"]]:
        r
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(f"Expected a string, got {value}")

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}.")

        return elements


@dataclass
class FunctionFormatter(StringFormatter):
    def __post_init__(self):
        super().__post_init__()
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        content: str = kwargs.pop("content")
        regex = re.compile(r"<think>(.*)</think>", re.DOTALL)
        thought = re.search(regex, content)
        if thought:
            content = content.replace(thought.group(0), "")

        functions: list[FunctionCall] = []
        try:
            tool_calls = json.loads(content)
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                functions.append(
                    FunctionCall(tool_call["name"], json.dumps(tool_call["arguments"], ensure_ascii=False))
                )

        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in function message: {str([content])}.")

        function_str = self.tool_utils.function_formatter(functions)
        if thought:
            function_str = thought.group(0) + function_str

        return super().apply(content=function_str)


@dataclass
class ToolFormatter(Formatter):
    def __post_init__(self):
        self.tool_utils = get_tool_utils(self.tool_format)

    @override
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            return [self.tool_utils.tool_formatter(tools) if len(tools) != 0 else ""]
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in tool description: {str([content])}.")

    @override
    def extract(self, content: str) -> Union[str, list["FunctionCall"]]:
        return self.tool_utils.tool_extractor(content)
