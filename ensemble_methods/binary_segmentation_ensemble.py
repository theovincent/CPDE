r"""Binary segmentation ensemble."""
from cProfile import label
from functools import lru_cache

import numpy as np

from ruptures.detection.binseg import Binseg

from ensemble_methods.cost_ensemble import CostEnsemble


class BinsegEnsemble(Binseg):

    """Binary segmentation."""

    def __init__(
        self,
        models=["l2"],
        min_size=2,
        jump=5,
        params={},
        scale_aggregation=lambda array: array,
    ):
        """Initialize a Binseg instance.

        Args:
            models (list[str], optional): segment model, [["l1", "l2"], ["rbf"]].
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5 samples.
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        self.model_names = models
        self.cost = CostEnsemble(models, params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None
        self.signal = None
        self.scale_aggregation = scale_aggregation

    @lru_cache(maxsize=None)
    def _single_bkp(self, start, end):
        """Return the optimal breakpoint of [start:end] (if it exists)."""
        segment_cost = self.cost.error(start, end)
        if any(np.isinf(segment_cost)) and any(segment_cost < 0):  # if cost is -inf
            return None, 0
        gain_list = list()
        # ---------NEW PART----------from here
        for bkp in range(start, end, self.jump):
            if bkp - start > self.min_size and end - bkp > self.min_size:
                gain = (
                    segment_cost
                    - self.cost.error(start, bkp)
                    - self.cost.error(bkp, end)
                )
                gain_list.append((gain, bkp))
        try:
            scores = [i[0] for i in gain_list]
            gain, bkp = max(
                np.array(
                    [
                        -1 * self.scale_aggregation(-1 * np.array(scores)),
                        np.array(gain_list)[:, 1],
                    ]
                ).T,
                key=lambda x: x[0],
            )
        # ---------NEW PART----------till here
        except ValueError:  # if empty sub_sampling
            return None, 0
        return bkp, gain
