r"""Window ensemble-based change point detection"""


import numpy as np

from scipy.signal import argrelmax
from ruptures.utils import unzip
from ruptures.detection.window import Window

from ruptures_changing.aggregations import selected_aggregation



class WindowEnsemble(Window):

    """Window sliding method."""

    def __init__(
        self, width=100, model="l2", custom_cost=None, min_size=2, jump=5, params=None, ensembling=1
    ):
        """Instanciate with window length.

        Args:
            width (int, optional): window length. Defaults to 100 samples.
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if `custom_cost` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.`
        """
        super(WindowEnsemble, self).__init__(width, model, custom_cost, min_size, jump, params)
        self.ensembling = ensembling

    def _seg(self, n_bkps=None, **kwargs):
        """Sequential peak search.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.

        Returns:
            list: breakpoint index list
        """

        # initialization
        bkps = [self.n_samples]
        stop = False
        # peak search
        peak_inds_shifted, = argrelmax(self.score,
                                       order=max(self.width, self.min_size) // (
                                           2 * self.jump),
                                       mode="wrap")
        gains = np.take(self.score, peak_inds_shifted)
        peak_inds_arr = np.take(self.inds, peak_inds_shifted)
        # sort according to score value
        _, peak_inds = unzip(sorted(zip(gains, peak_inds_arr)))
        peak_inds = list(peak_inds)

        while not stop:
            stop = True

            try:
                # index with maximum score
                bkp = peak_inds.pop()
            except IndexError:  # peak_inds is empty
                break

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False

            if not stop:
                bkps.append(bkp)
                bkps.sort()

        return bkps

    def fit(self, signal) -> "Window":
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples, _ = self.signal.shape
        # indexes
        self.inds = np.arange(self.n_samples, step=self.jump)
        # delete borders
        keep = (self.inds >= self.width // 2) & (
            self.inds < self.n_samples - self.width // 2
        )
        self.inds = self.inds[keep]
        self.cost.fit(signal)
        # compute score
        score = list()
        for k in self.inds:
            start, end = k - self.width // 2, k + self.width // 2
        #---------NEW PART----------from here
            gain = np.array(self.cost.error(start, end))
            gain -= np.array(self.cost.error(start, k)) + np.array(self.cost.error(k, end))
            score.append(gain)
        self.score = selected_aggregation(self.ensembling)(np.array(score))
        #---------NEW PART----------till here
        return self