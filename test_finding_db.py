import numpy as np
import matplotlib.pyplot as plt
sigma = 0.15 

r_1 = 1 #kpc
r_2 = 10 #kpc
db = 0.01

def z(r, b):
    return r*np.sin(b)

def height_distribution(z): # z is the height above the Galactic plane
    """
    Args:
        z: height above the Galactic plane
    Returns:
        height distribution of the disk at a height z above the Galactic plane
    """
    return np.exp(-0.5 * z**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


z_11 = z(r_1, db)
z_12 = z(r_1, 2*db)
dz_1 = z_12 - z_11

z_21 = z(r_2, db)
z_22 = z(r_2, 2*db)
dz_2 = z_22 - z_21

print("dz_1: ", dz_1)
print("dz_2: ", dz_2)


zs = np.arange(0, 5, 0.01)
density = height_distribution(zs)
plt.plot(zs, density)
plt.show()
plt.close()