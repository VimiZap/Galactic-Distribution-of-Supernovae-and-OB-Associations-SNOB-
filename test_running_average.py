import numpy as np
import matplotlib.pyplot as plt

# generate the set of coordinates
dr = 0.01   # increments in dr (kpc):
dl = 0.2   # increments in dl (degrees):
db = 0.1   # increments in db (degrees):
# latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
#latitude_range = np.radians(3.5)
latitudes = np.radians(np.arange(-0.5, 0.5 + db, db))
print("latidues shape: ", latitudes.shape)
# np.array with values for galactic longitude l in radians.
l1 = np.arange(180, 0, -dl)
l2 = np.arange(360, 180, -dl)
longitudes = np.radians(np.concatenate((l1, l2)))
print("longitudes shape: ", longitudes.shape)
# np.array with values for distance from the Sun to the star/ a point in the Galaxy
radial_distances = np.arange(dr, 7.6 + 10 + 5 + dr, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
print("radial_distances shape", radial_distances.shape)

# // floors the number, i.e. rounds down to the nearest integer
rng = np.random.default_rng()
array_initial = rng.integers(low=1, high=25, size=(100))
delta_l_halves = (5/dl)//2

array_initial_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
delta_test = 5
print("len array test: ", len(array_initial_test))
 
def running_average(data, window_size):
   array_running_averaged = []
   delta = (window_size)//2
   for i in range(len(data)):
      if i-delta < 0:
         val = np.sum(data[-delta + i:]) + np.sum(data[:delta + i + 1])
         array_running_averaged.append(val)
      elif i+delta >= len(data):
         val = np.sum(data[i-delta:]) + np.sum(data[:delta + i - len(data) + 1])
         array_running_averaged.append(val)
      else:
         array_running_averaged.append(np.sum(data[i-delta:i+delta + 1]))
   return np.array(array_running_averaged) / window_size

array_running_averaged = running_average(array_initial, delta_test)
plt.plot(array_initial, label="initial array")
plt.plot(array_running_averaged, label="running averaged array")
plt.legend()
plt.show()
#print(array_running_averaged) 