import numpy as np

def unsqueeze(array):
  if isinstance(array, list):
    return array
  else:
    return [array]

def softmax(w, t=1.0, axis=None):
  w = np.array(w) / t
  e = np.exp(w - np.amax(w, axis=axis, keepdims=True))
  dist = e / np.sum(e, axis=axis, keepdims=True)
  return dist










