import numpy as np
import matplotlib.pyplot as plt 
# generate the set of coordinates
dr = 0.01   # increments in dr (kpc):
dl = 0.2   # increments in dl (degrees):
db = 0.2   # increments in db (degrees):
# latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
latitudes = np.radians(np.arange(-3.5, 3.5 + db, db))
print("latidues shape: ", latitudes.shape)
print("latitudes: ", latitudes)
# np.array with values for galactic longitude l in radians.
l1 = np.arange(180, 0, -dl)
l2 = np.arange(360, 180, -dl)
longitudes = np.radians(np.concatenate((l1, l2)))
print("longitudes shape: ", longitudes.shape)
print("longitudes: ", longitudes)
long = len(longitudes)
lat = len(latitudes)

long_grid, lat_grid = np.meshgrid(longitudes, latitudes, indexing='ij')
test_data = np.zeros((long + 1, lat + 1))
print("test_data shape: ", test_data.shape)
# Create test data with a gradient
test_data[1:, 1:] = 1 - long_grid / long  # Gradient along longitudes
test_data[1:, 1:] *= 1 - lat_grid / lat   # Multiply by gradient along latitudes

test_data[1:, 0] = longitudes
test_data[0, 1:] = latitudes
print(test_data)

np.savetxt('test_data.txt', test_data)

skydata = np.loadtxt('test_data.txt')
print(skydata)
print(skydata.shape)
# Create coordinate grids
long_grid, lat_grid = np.meshgrid(np.linspace(0, 100, len(skydata[1:, 0])), skydata[0, 1:], indexing='ij')

plt.pcolormesh(long_grid, lat_grid, skydata[1:, 1:], shading='auto')  
plt.colorbar()
# Redefine the x-axis labels to match the values in longitudes
x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
plt.xticks(np.linspace(0, 100, 13), x_ticks)
plt.xlabel('Galactic Longitude (degrees)')
plt.ylabel('Galactic Latitude (degrees)')
plt.show()
