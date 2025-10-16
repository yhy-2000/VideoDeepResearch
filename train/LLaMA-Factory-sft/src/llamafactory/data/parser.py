
import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

from huggingface_hub import hf_hub_download

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope, use_openmind


@dataclass
class DatasetAttr:
    r

    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file"]
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    ranking: bool = False
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    audios: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    messages: Optional[str] = "conversations"
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))

    def join(self, attr: dict[str, Any]) -> None:
        self.set_attr("formatting", attr, default="alpaca")
        self.set_attr("ranking", attr, default=False)
        self.set_attr("subset", attr)
        self.set_attr("split", attr, default="train")
        self.set_attr("folder", attr)
        self.set_attr("num_samples", attr)

        if "columns" in attr:
            column_names = ["prompt", "query", "response", "history", "messages", "system", "tools"]
            column_names += ["images", "videos", "audios", "chosen", "rejected", "kto_tag"]
            for column_name in column_names:
                self.set_attr(column_name, attr["columns"])

        if "tags" in attr:
            tag_names = ["role_tag", "content_tag"]
            tag_names += ["user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"]
            for tag in tag_names:
                self.set_attr(tag, attr["tags"])


def get_dataset_list(dataset_names: Optional[list[str]], dataset_dir: str) -> list["DatasetAttr"]:
    r
    if dataset_names is None:
        dataset_names = []

    if dataset_dir == "ONLINE":
        dataset_info = None
    else:
        if dataset_dir.startswith("REMOTE:"):
            config_path = hf_hub_download(repo_id=dataset_dir[7:], filename=DATA_CONFIG, repo_type="dataset")
        else:
            config_path = os.path.join(dataset_dir, DATA_CONFIG)

        try:
            with open(config_path) as f:
                dataset_info = json.load(f)
        except Exception as err:
            if len(dataset_names) != 0:
                raise ValueError(f"Cannot open {config_path} due to {str(err)}.")

            dataset_info = None

    dataset_list: list[DatasetAttr] = []
    for name in dataset_names:
        if dataset_info is None:
            load_from = "ms_hub" if use_modelscope() else "om_hub" if use_openmind() else "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]
        has_om_url = "om_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url or has_om_url:
            if has_ms_url and (use_modelscope() or not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            elif has_om_url and (use_openmind() or not has_hf_url):
                dataset_attr = DatasetAttr("om_hub", dataset_name=dataset_info[name]["om_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        elif "cloud_file_name" in dataset_info[name]:
            dataset_attr = DatasetAttr("cloud_file", dataset_name=dataset_info[name]["cloud_file_name"])
        else:
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.join(dataset_info[name])
        dataset_list.append(dataset_attr)

    return dataset_list
