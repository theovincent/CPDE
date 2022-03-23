r"""Dynamic programming ensemble"""
from functools import lru_cache
import numpy as np

from ruptures.utils import sanity_check
from ruptures.detection.dynp import Dynp

from ruptures_changing.aggregations import selected_aggregation



class DynpEnsemble(Dynp):

    """Find optimal change points using dynamic programming.

    Given a segment model, it computes the best partition for which the
    sum of errors is minimum.
    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None, scale_aggregation=lambda array: array):
        """Creates a Dynp instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        super(DynpEnsemble, self).__init__(model, custom_cost, min_size, jump, params)
        self.scale_aggregation = scale_aggregation

    @lru_cache(maxsize=None)
    def _seg(self, start, end, n_bkps):
        """Recurrence to find the optimal partition of signal[start:end].

        This method is to be memoized and then used.

        Args:
            start (int): start of the segment (inclusive)
            end (int): end of the segment (exclusive)
            n_bkps (int): number of breakpoints

        Returns:
            dict: {(start, end): cost value, ...}
        """
        jump, min_size = self.jump, self.min_size

        if n_bkps == 0:
            cost = self.cost.error(start, end)
            return {(start, end): cost}
        elif n_bkps > 0:
            # Let's fill the list of admissible last breakpoints
            multiple_of_jump = (k for k in range(start, end) if k % jump == 0)
            admissible_bkps = list()
            for bkp in multiple_of_jump:
                n_samples = bkp - start
                # first check if left subproblem is possible
                if sanity_check(
                    n_samples=n_samples,
                    n_bkps=n_bkps - 1,
                    jump=jump,
                    min_size=min_size,
                ):
                    # second check if the right subproblem has enough points
                    if end - bkp >= min_size:
                        admissible_bkps.append(bkp)

            assert (
                len(admissible_bkps) > 0
            ), "No admissible last breakpoints found.\
             start, end: ({},{}), n_bkps: {}.".format(
                start, end, n_bkps
            )

            # Compute the subproblems
            sub_problems = list()
            for bkp in admissible_bkps:
                left_partition = self.seg(start, bkp, n_bkps - 1)
                right_partition = self.seg(bkp, end, 0)
                tmp_partition = dict(left_partition)
                tmp_partition[(bkp, end)] = right_partition[(bkp, end)]
                sub_problems.append(tmp_partition)

            #---------NEW PART----------from here
            #number of cost functions used in an ensemble
            number_of_costs = len(tmp_partition[(bkp, end)])
            #ensembling
            array = np.array([[sum([i[cost_number] for i in sub_pr.values()]) for cost_number in range(number_of_costs)] for sub_pr in sub_problems])

            # Find the optimal partition
            return sub_problems[np.argmin(self.scale_aggregation(array))]
            #---------NEW PART----------till here
