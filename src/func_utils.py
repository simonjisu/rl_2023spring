import numpy as np
import math



def tanh(vals):
  """
  tanh function
  inputs:
    vals      1d array
  """
  return np.array([math.tanh(x) for x in vals])

def min_max(vals, is_tanh_like=False):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  range_val = max_val - min_val
  if is_tanh_like:
    return 2 * ((vals - min_val) / (range_val + 1e-10)) - 1
  else:
    return (vals - min_val) / (range_val + 1e-10)

def sigmoid(xs):
  """
  sigmoid function
  inputs:
    xs      1d array
  """
  return np.array([1 / (1 + math.exp(-x)) for x in xs])