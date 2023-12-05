import numpy as np
r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used

x, y, z = -10, -1000000, 0
long = (np.arctan2(y - r_s, x) + np.pi/2) % (2 * np.pi)
print(np.degrees(long))

empty = np.array([])
empty = np.concatenate((empty, np.array([1, 2, 3])))
empty = np.concatenate((empty, np.array([1, 22, 3])))
print(empty)