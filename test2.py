import numpy as np

# Create arrays of size (1000, 100) for x and y values
n = 1000
m = 100

x_values = np.random.rand(n, 1)  # Replace with your actual x values
y_values = np.random.rand(1, m)  # Replace with your actual y values

# Combine x and y values into a single array of size (1000, 100, 2)
combined_array = np.stack((x_values[:, :, np.newaxis], y_values[np.newaxis, :, :]), axis=2)
print(combined_array.shape)
# 'combined_array' is now a new array of size (1000, 100, 2)
