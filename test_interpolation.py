import numpy as np
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
grid_x, grid_y = grid_x[:-2], grid_y[:-2]
grid_x1, grid_y1 = grid_x.flatten(), grid_y.flatten()
print(grid_x.shape, grid_y.shape, grid_x1.shape, grid_y1.shape)
grid_x1 = grid_x1[:-2]
grid_y1 = grid_y1[:-2]
print(grid_x.shape, grid_y.shape, grid_x1.shape, grid_y1.shape)


rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:,0], points[:,1])
#values = values.flatten()

from scipy.interpolate import griddata
grid_z01 = griddata(points, values, (grid_x1, grid_y1), method='nearest')
grid_z11 = griddata(points, values, (grid_x1, grid_y1), method='linear')
grid_z21 = griddata(points, values, (grid_x1, grid_y1), method='cubic')

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

print(grid_z0.flatten()[:-2] == grid_z01)
print(grid_z01.shape, grid_z0.shape, grid_z0.size, grid_z0.flatten().shape)


import matplotlib.pyplot as plt
plt.subplot(221)
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()