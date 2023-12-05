import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 1)
y = np.arange(10, 20, 1)
z = np.arange(20, 30, 1)

x_target = 3 
y_target = 5
z_target = 9
print("targeted values: ", x[x_target], y[y_target], z[z_target])
x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
x_grid, y_grid, z_grid = x_grid.ravel(), y_grid.ravel(), z_grid.ravel()

#target_z = z_grid[lon_index + lat_index * len(longitudes) + radial_index * len(longitudes) * len(latitudes)] #rad, long, lat
#rad_index * len(longitudes) * len(latitudes) + long_index * len(latitudes) + lat_index

index_old = y_target + z_target * len(y) + x_target * len(y) * len(z)
index = x_target*len(y)*len(z) + y_target * len(z) + z_target
x_val = x_grid[index]
y_val = y_grid[index]
z_val = z_grid[index]
print("obtained values: ", x_val, y_val, z_val)

xx = np.arange(0, 20, 1)
yy = np.sin(xx)
plt.plot(xx, yy)
plt.suptitle("Monte Carlo simulation of temporal clustering of SNPs")
plt.title("Made with 100000 associations")
plt.show()