import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

rng = np.random.default_rng()
arr = rng.integers(low=1, high=10, size=(3, 2, 4))
rad_densities = arr/np.sum(arr, axis=0, keepdims=True)
print(rad_densities)
print(np.sum(rad_densities[:, 0, 2]))
print(np.sum(rad_densities))
print("###################")
print(arr)
print(np.sum(arr, axis = 0))
print(np.sum(arr[0], axis = 0))
print(np.sum(arr, axis = 1))
print(np.sum(arr, axis = 2))