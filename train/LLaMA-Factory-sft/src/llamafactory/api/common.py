
import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pydantic import BaseModel


def dictify(data: "BaseModel") -> dict[str, Any]:
    try:
        return data.model_dump(exclude_unset=True)
    except AttributeError:
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:
        return data.json(exclude_unset=True, ensure_ascii=False)
