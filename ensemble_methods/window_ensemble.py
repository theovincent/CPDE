r"""Window ensemble-based change point detection"""

import numpy as np

from ruptures.detection.window import Window

from ensemble_methods.cost_ensemble import CostEnsemble


class WindowEnsemble(Window):

    """Window sliding method."""

    def __init__(
        self,
        width=100,
        models=["l2"],
        min_size=2,
        jump=5,
        params={},
        scale_aggregation=None,
    ):
        """Instanciate with window length.

        Args:
            width (int, optional): window length. Defaults to 100 samples.
            models (list[str], optional): segment model, [["l1", "l2"], ["rbf"]].
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.`
        """
        self.model_names = models
        self.inds = None
        self.width = 2 * (width // 2)
        self.cost = CostEnsemble(models, params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None
        self.signal = None
        self.score = list()
        self.scale_aggregation = scale_aggregation

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
            # ---------NEW PART----------from here
            gain = self.cost.error(start, end)
            gain -= self.cost.error(start, k) + self.cost.error(k, end)
            score.append(gain)
        self.score = -1 * self.scale_aggregation(-1 * score)
        # ---------NEW PART----------till here
        return self
