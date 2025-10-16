
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        chosen_path: Optional[list[dict[str, str]]] = None,
        rejected_path: Optional[list[dict[str, str]]] = None,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        
        if chosen_path is not None and rejected_path is not None:
            chosen_messages = self.template.mm_plugin.process_messages(
                chosen_path, images, videos, audios, self.processor
            )
            rejected_messages = self.template.mm_plugin.process_messages(
                rejected_path, images, videos, audios, self.processor
            )
            
            chosen_input_ids, chosen_labels = self.template.encode_path(
                self.tokenizer, chosen_messages, system, tools
            )
            rejected_input_ids, rejected_labels = self.template.encode_path(
                self.tokenizer, rejected_messages, system, tools
            )
            
            if self.template.efficient_eos:
                chosen_input_ids += [self.tokenizer.eos_token_id]
                chosen_labels += [self.tokenizer.eos_token_id]
                rejected_input_ids += [self.tokenizer.eos_token_id]
                rejected_labels += [self.tokenizer.eos_token_id]
                
            chosen_input_ids, chosen_labels = self.template.mm_plugin.process_token_ids(
                chosen_input_ids, chosen_labels, images, videos, audios, self.tokenizer, self.processor
            )
            rejected_input_ids, rejected_labels = self.template.mm_plugin.process_token_ids(
                rejected_input_ids, rejected_labels, images, videos, audios, self.tokenizer, self.processor
            )
            
            chosen_input_ids = chosen_input_ids[:self.data_args.cutoff_len]
            chosen_labels = chosen_labels[:self.data_args.cutoff_len]
            rejected_input_ids = rejected_input_ids[:self.data_args.cutoff_len]
            rejected_labels = rejected_labels[:self.data_args.cutoff_len]
            
            return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels
        
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        
        if len(prompt) > 1:
            chosen_encoded_pairs = self.template.encode_multiturn(self.tokenizer, chosen_messages, system, tools)
            rejected_encoded_pairs = self.template.encode_multiturn(self.tokenizer, rejected_messages, system, tools)
            
            prompt_ids = []
            chosen_ids = []
            rejected_ids = []
            
            for i, (source_ids, target_ids) in enumerate(chosen_encoded_pairs):
                if i < len(chosen_encoded_pairs) - 1:
                    prompt_ids.extend(source_ids + target_ids)
                else:
                    prompt_ids.extend(source_ids)
                    chosen_ids.extend(target_ids)
            
            for i, (source_ids, target_ids) in enumerate(rejected_encoded_pairs):
                if i == len(rejected_encoded_pairs) - 1:
                    rejected_ids.extend(target_ids)
                    
        else:
            prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
            _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"]) if "_prompt" in examples else len(examples.get("_chosen_path", []))):
            if "_chosen_path" in examples and "_rejected_path" in examples:
                chosen_path = examples["_chosen_path"][i]
                rejected_path = examples["_rejected_path"][i]
                
                if not chosen_path or not rejected_path:
                    logger.warning_rank0(f"Dropped invalid path-based example: chosen_path={chosen_path}, rejected_path={rejected_path}")
                    continue
                
                chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                    prompt=[],
                    response=[],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                    chosen_path=chosen_path,
                    rejected_path=rejected_path,
                )
            else:
                if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                    logger.warning_rank0(
                        "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                    )
                    continue

                chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )
            
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
