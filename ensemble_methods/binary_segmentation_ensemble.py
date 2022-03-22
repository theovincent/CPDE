r"""Binary segmentation ensemble."""
from functools import lru_cache

import numpy as np

from ruptures.detection.binseg import Binseg

from ruptures_changing.aggregations import selected_aggregation


class BinsegEnsemble(Binseg):

    """Binary segmentation."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None, ensembling=1):
        """Initialize a Binseg instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf",...]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5 samples.
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        super(BinsegEnsemble, self).__init__(model, custom_cost, min_size, jump, params)
        self.ensembling = ensembling


    @lru_cache(maxsize=None)
    def _single_bkp(self, start, end):
        """Return the optimal breakpoint of [start:end] (if it exists)."""
        segment_cost = self.cost.error(start, end)
        gain_list = list()
        #---------NEW PART----------from here
        for bkp in range(start, end, self.jump):
            if bkp - start > self.min_size and end - bkp > self.min_size:
                gain = segment_cost - \
                    np.array(self.cost.error(start, bkp)) - np.array(self.cost.error(bkp, end))
                gain_list.append((gain, bkp))
        try:
            scores = [i[0] for i in gain_list]
            gain, bkp = max(np.array([(-1)*selected_aggregation(self.ensembling)(np.array(scores)*(-1)), np.array(gain_list)[:,1]]).T, key=lambda x: x[0])
        #---------NEW PART----------till here
        except ValueError:  # if empty sub_sampling
            return None, 0
        return bkp, gain