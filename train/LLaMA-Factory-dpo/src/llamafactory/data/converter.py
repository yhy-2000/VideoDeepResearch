
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ..extras import logging
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> Optional[list["MediaType"]]:
        r
        if medias is None:
            return None
        elif not isinstance(medias, list):
            medias = [medias]
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"]:
            if isinstance(medias[0], str):
                for i in range(len(medias)):
                    media_path = os.path.join(self.data_args.media_dir, medias[i])
                    if os.path.isfile(media_path):
                        medias[i] = media_path
                    else:
                        logger.warning_rank0_once(
                            f"Media {medias[i]} does not exist in `media_dir`. Use original path."
                        )
            elif isinstance(medias[0], list):
                for i in range(len(medias)):
                    for j in range(len(medias[i])):
                        media_path = os.path.join(self.data_args.media_dir, medias[i][j])
                        if os.path.isfile(media_path):
                            medias[i][j] = media_path
                        else:
                            logger.warning_rank0_once(
                                f"Media {medias[i][j]} does not exist in `media_dir`. Use original path."
                            )

        return medias

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], (str, list))
            and isinstance(example[self.dataset_attr.rejected], (str, list))
        ):
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            
            if isinstance(chosen, list) and isinstance(rejected, list):
                chosen_messages = []
                rejected_messages = []
                
                for msg in chosen:
                    if isinstance(msg, dict) and "from" in msg and "value" in msg:
                        role_map = {
                            "human": Role.USER.value,
                            "gpt": Role.ASSISTANT.value,
                            "user": Role.USER.value,
                            "assistant": Role.ASSISTANT.value,
                            "system": Role.SYSTEM.value
                        }
                        role = role_map.get(msg["from"], msg["from"])
                        chosen_messages.append({
                            "role": role,
                            "content": msg["value"]
                        })
                
                for msg in rejected:
                    if isinstance(msg, dict) and "from" in msg and "value" in msg:
                        role_map = {
                            "human": Role.USER.value,
                            "gpt": Role.ASSISTANT.value,
                            "user": Role.USER.value,
                            "assistant": Role.ASSISTANT.value,
                            "system": Role.SYSTEM.value
                        }
                        role = role_map.get(msg["from"], msg["from"])
                        rejected_messages.append({
                            "role": role,
                            "content": msg["value"]
                        })
                
                if len(chosen_messages) > 0 and chosen_messages[-1]["role"] == Role.ASSISTANT.value:
                    prompt = chosen_messages[:-1]
                    chosen_response = chosen_messages[-1]
                else:
                    prompt = chosen_messages
                    chosen_response = {"role": Role.ASSISTANT.value, "content": ""}
                
                if len(rejected_messages) > 0 and rejected_messages[-1]["role"] == Role.ASSISTANT.value:
                    rejected_response = rejected_messages[-1]
                else:
                    rejected_response = {"role": Role.ASSISTANT.value, "content": ""}
                
                response = [chosen_response, rejected_response]
            else:
                response = [
                    {"role": Role.ASSISTANT.value, "content": chosen},
                    {"role": Role.ASSISTANT.value, "content": rejected},
                ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:
            response = []
        


        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example['conversations']
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0 and not (
                isinstance(example.get(self.dataset_attr.chosen), list) and 
                isinstance(example.get(self.dataset_attr.rejected), list)
            )
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], (dict, list))
            and isinstance(example[self.dataset_attr.rejected], (dict, list))
        ):
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            
            if isinstance(chosen, list) and isinstance(rejected, list):
                chosen_messages = []
                rejected_messages = []
                
                for msg in chosen:
                    if isinstance(msg, dict) and "from" in msg and "value" in msg:
                        role_map = {
                            "human": Role.USER.value,
                            "gpt": Role.ASSISTANT.value,
                            "user": Role.USER.value,
                            "assistant": Role.ASSISTANT.value,
                            "system": Role.SYSTEM.value
                        }
                        role = role_map.get(msg["from"], msg["from"])
                        chosen_messages.append({
                            "role": role,
                            "content": msg["value"]
                        })
                
                for msg in rejected:
                    if isinstance(msg, dict) and "from" in msg and "value" in msg:
                        role_map = {
                            "human": Role.USER.value,
                            "gpt": Role.ASSISTANT.value,
                            "user": Role.USER.value,
                            "assistant": Role.ASSISTANT.value,
                            "system": Role.SYSTEM.value
                        }
                        role = role_map.get(msg["from"], msg["from"])
                        rejected_messages.append({
                            "role": role,
                            "content": msg["value"]
                        })
                
                if len(chosen_messages) > 0 and chosen_messages[-1]["role"] == Role.ASSISTANT.value:
                    prompt = chosen_messages[:-1]
                    chosen_response = chosen_messages[-1:]
                else:
                    prompt = chosen_messages
                    chosen_response = [{"role": Role.ASSISTANT.value, "content": ""}]
                
                if len(rejected_messages) > 0 and rejected_messages[-1]["role"] == Role.ASSISTANT.value:
                    rejected_response = rejected_messages[-1:]
                else:
                    rejected_response = [{"role": Role.ASSISTANT.value, "content": ""}]
                
                response = [
                    chosen_response[0],
                    rejected_response[0]
                ]
            else:
                if (
                    chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                    or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
                ):
                    logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                    broken_data = True

                prompt = aligned_messages
                response = [
                    {
                        "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                        "content": chosen[self.dataset_attr.content_tag],
                    },
                    {
                        "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                        "content": rejected[self.dataset_attr.content_tag],
                    },
                ]
        else:
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class PathDPODatasetConverter(DatasetConverter):
    r
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        
        system = ""
        if self.dataset_attr.system and example[self.dataset_attr.system]:
            system = example[self.dataset_attr.system]
        
        chosen_conversation = []
        rejected_conversation = []
        
        if (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], list)
            and isinstance(example[self.dataset_attr.rejected], list)
        ):
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            
            for msg in chosen:
                if isinstance(msg, dict) and "from" in msg and "value" in msg:
                    role_map = {
                        "human": Role.USER.value,
                        "gpt": Role.ASSISTANT.value,
                        "user": Role.USER.value,
                        "assistant": Role.ASSISTANT.value,
                        "system": Role.SYSTEM.value
                    }
                    role = role_map.get(msg["from"], msg["from"])
                    chosen_conversation.append({
                        "role": role,
                        "content": msg["value"]
                    })
            
            for msg in rejected:
                if isinstance(msg, dict) and "from" in msg and "value" in msg:
                    role_map = {
                        "human": Role.USER.value,
                        "gpt": Role.ASSISTANT.value,
                        "user": Role.USER.value,
                        "assistant": Role.ASSISTANT.value,
                        "system": Role.SYSTEM.value
                    }
                    role = role_map.get(msg["from"], msg["from"])
                    rejected_conversation.append({
                        "role": role,
                        "content": msg["value"]
                    })
                    
        elif (
            self.dataset_attr.messages and 
            isinstance(example[self.dataset_attr.messages], list) and
            len(example[self.dataset_attr.messages]) > 0
        ):
            messages = example[self.dataset_attr.messages]
            
            if (
                self.dataset_attr.system_tag
                and len(messages) != 0
                and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
            ):
                system = messages[0][self.dataset_attr.content_tag]
                messages = messages[1:]
            
            base_conversation = []
            for message in messages:
                role = tag_mapping.get(message[self.dataset_attr.role_tag], message[self.dataset_attr.role_tag])
                base_conversation.append({
                    "role": role,
                    "content": message[self.dataset_attr.content_tag]
                })
            
            chosen_conversation = base_conversation[:]
            rejected_conversation = base_conversation[:]
            
            if (
                self.dataset_attr.ranking
                and self.dataset_attr.chosen in example
                and self.dataset_attr.rejected in example
            ):
                chosen_resp = example[self.dataset_attr.chosen]
                rejected_resp = example[self.dataset_attr.rejected]
                
                if isinstance(chosen_resp, dict):
                    chosen_role = tag_mapping.get(chosen_resp[self.dataset_attr.role_tag], chosen_resp[self.dataset_attr.role_tag])
                    chosen_conversation.append({
                        "role": chosen_role,
                        "content": chosen_resp[self.dataset_attr.content_tag]
                    })
                    
                if isinstance(rejected_resp, dict):
                    rejected_role = tag_mapping.get(rejected_resp[self.dataset_attr.role_tag], rejected_resp[self.dataset_attr.role_tag])
                    rejected_conversation.append({
                        "role": rejected_role,
                        "content": rejected_resp[self.dataset_attr.content_tag]
                    })
        else:
            chosen_conversation = []
            rejected_conversation = []
        
        output = {
            "_chosen_path": chosen_conversation,
            "_rejected_path": rejected_conversation,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "path_dpo": PathDPODatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    r
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    return dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )
