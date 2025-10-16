
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeAccuracy:
    r

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        chosen_scores, rejected_scores = numpify(eval_preds.predictions[0]), numpify(eval_preds.predictions[1])
        if not chosen_scores.shape:
            self.score_dict["accuracy"].append(chosen_scores > rejected_scores)
        else:
            for i in range(len(chosen_scores)):
                self.score_dict["accuracy"].append(chosen_scores[i] > rejected_scores[i])

        if compute_result:
            return self._dump()
