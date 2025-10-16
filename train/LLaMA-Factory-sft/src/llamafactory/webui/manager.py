
from collections.abc import Generator
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gradio.components import Component


class Manager:
    r

    def __init__(self) -> None:
        self._id_to_elem: dict[str, Component] = {}
        self._elem_to_id: dict[Component, str] = {}

    def add_elems(self, tab_name: str, elem_dict: dict[str, "Component"]) -> None:
        r
        for elem_name, elem in elem_dict.items():
            elem_id = f"{tab_name}.{elem_name}"
            self._id_to_elem[elem_id] = elem
            self._elem_to_id[elem] = elem_id

    def get_elem_list(self) -> list["Component"]:
        r
        return list(self._id_to_elem.values())

    def get_elem_iter(self) -> Generator[tuple[str, "Component"], None, None]:
        r
        for elem_id, elem in self._id_to_elem.items():
            yield elem_id.split(".")[-1], elem

    def get_elem_by_id(self, elem_id: str) -> "Component":
        r
        return self._id_to_elem[elem_id]

    def get_id_by_elem(self, elem: "Component") -> str:
        r
        return self._elem_to_id[elem]

    def get_base_elems(self) -> set["Component"]:
        r
        return {
            self._id_to_elem["top.lang"],
            self._id_to_elem["top.model_name"],
            self._id_to_elem["top.model_path"],
            self._id_to_elem["top.finetuning_type"],
            self._id_to_elem["top.checkpoint_path"],
            self._id_to_elem["top.quantization_bit"],
            self._id_to_elem["top.quantization_method"],
            self._id_to_elem["top.template"],
            self._id_to_elem["top.rope_scaling"],
            self._id_to_elem["top.booster"],
        }
