
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Union


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from vllm import AsyncLLMEngine

    from ..data import Template
    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput
    from ..extras.constants import EngineName
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]


class BaseEngine(ABC):
    r

    name: "EngineName"
    model: Union["PreTrainedModel", "AsyncLLMEngine"]
    tokenizer: "PreTrainedTokenizer"
    can_generate: bool
    template: "Template"
    generating_args: dict[str, Any]

    @abstractmethod
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        r
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        r
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r
        ...

    @abstractmethod
    async def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r
        ...
