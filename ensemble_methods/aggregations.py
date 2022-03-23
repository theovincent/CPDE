import numpy as np

SCALING_AGGREGATION_PAPER = {
    "Min_MinMax": lambda array: min_(minmax(array)),
    "Min_Znorm": lambda array: min_(znorm(array)),
    "Min_MinAbs": lambda array: min_(minabs(array)),
    "Min_Rank": lambda array: min_(rank(array)),
    "Sum_MinMax": lambda array: sum_(minmax(array)),
    "Sum_Znorm": lambda array: sum_(znorm(array)),
    "Sum_MinAbs": lambda array: sum_(minabs(array)),
    "Sum_Rank": lambda array: sum_(rank(array)),
    "WeightedSum_MinMax": lambda array: weightedsum(minmax(array), array),
    "WeightedSum_Znorm": lambda array: weightedsum(znorm(array), array),
    "WeightedSum_MinAbs": lambda array: weightedsum(minabs(array), array),
    "WeightedSum_Rank": lambda array: weightedsum(rank(array), array),
    "ThresholdSum_MinMax": lambda array: thresholdsum(minmax(array)),
    "ThresholdSum_Znorm": lambda array: thresholdsum(znorm(array)),
    "ThresholdSum_MinAbs": lambda array: thresholdsum(minabs(array)),
    "ThresholdSum_Rank": lambda array: thresholdsum(rank(array)),
}

SCALING_AGGREGATION_INVESTIGATION = {
    "Min_MinMax": lambda array: min_(minmax(array)),
    "Min_Raw": lambda array: min_(array),
    "Max_MinMax": lambda array: max_(minmax(array)),
    "Max_Raw": lambda array: max_(array),
    "WeightedSum_MinMax": lambda array: weightedsum(minmax(array), array),
    "WeightedSum_Raw": lambda array: weightedsum(array, array),
}

# Scaling functions
def minmax(array):
    return (array - np.max(array, axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0))
    
def znorm(array):
    return (array - np.mean(array, axis=0)) / np.std(array, axis=0)
    
def minabs(array):
    return array / abs(np.min(array, axis=0) + 0.001)
    
def rank(array):
    order = array.argsort(axis=0)
    ranks = order.argsort(axis=0)
    return ranks

# Aggreation function
def min_(array):
    return np.min(array, axis=1).T

def sum_(array):
    return np.sum(array, axis=1).T

def weightedsum(array, raw_array):
    min_array = np.min(raw_array, axis=0)
    weights = (np.max(array, axis=0) - min_array) / (np.mean(array, axis=0) - min_array + 1e-8)

    return weights @ array.T

def thresholdsum(array):
    threshold = np.mean(array, axis=0)
    lower_threshold_indices = array > threshold
    array[lower_threshold_indices] = 0
    return np.sum(array, axis=1).T

def max_(array):
    return np.max(array, axis=1).T
    