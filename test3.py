import numpy as np

x = np.arange(0, 3, 1)
y = np.arange(3, 7, 1)
z = np.arange(10, 13, 1)
print(x, y, z)

x1, y1, z1 = np.meshgrid(x, y, z, indexing='ij')
coordinates = np.column_stack((x1.ravel(), y1.ravel(), z1.ravel()))
print(x1, y1, z1)
print(coordinates.shape)
print(coordinates)
print("sxdcfbhjnmk,l")
#print(coordinates[::0])
print(coordinates[:, 0])
print(coordinates[:, 1])
print(coordinates[:, 2])
#print(coordinates[::2])