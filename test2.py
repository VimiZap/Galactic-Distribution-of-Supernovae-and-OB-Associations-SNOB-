import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

rng = np.random.default_rng()
arr = rng.integers(low=1, high=10, size=(3, 2, 4))

print(arr)
print(np.sum(arr, axis = 0))
print(np.sum(arr, axis = 1))
print(np.sum(arr, axis = 2))