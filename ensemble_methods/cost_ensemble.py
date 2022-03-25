import numpy as np

from ruptures.costs import cost_factory
from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostEnsemble(BaseCost):    
    r"""Cost ensemble."""

    model = "ensemble"

    def __init__(self, models, params={}):
        """Initialize the object.

        Args:
            models (list[str], optional): segment model, [["l1", "l2"], ["rbf"]].
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        self.signal = None
        self.model_names = models
        self.costs = []
        self.min_size = 3
        for model in models:
            if params.get(model, None) is None:
                self.costs.append(cost_factory(model=model))
            else:
                self.costs.append(cost_factory(model=model, **params[model]))
            self.min_size = max(self.min_size, self.costs[-1].min_size)

    def fit(self, signal) -> "CostEnsemble":
        """Set parameters of the instance.
        Args:
            signal (array): signal of shape (n_samples, n_dims) or (n_samples,)
        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        for cost in self.costs:
            cost.fit(signal)
           
        return self
    
    
    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].
        Args:
            start (int): start of the segment
            end (int): end of the segment
        Returns:
            array: segment cost for each cost
        Raises:
            NotEnoughPoints: when the segment is too short (less than
                ``'min_size'`` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        
        return np.array([cost.error(start, end) for cost in self.costs])

    