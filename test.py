import numpy as np

a = np.array([[1, 2, 3], [4,5,6]])
b = a[:, np.newaxis]
c = a.T
""" print(a.shape, b.shape, c.shape)
print(a)
print(b.T)
print(c)
print(a[:, ::-1]) """

# Original 2D array
original_array = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])

# Create a new 2D array with negative values and reverse order
negative_array = -np.flipud(original_array)
print(negative_array)  
# Concatenate the two arrays along the second axis (columns)
transformed_array = np.concatenate((negative_array, original_array), axis=0)

# Output the transformed array
print(original_array)
print(original_array.T)
print(transformed_array)

array1 = np.random.rand(1039, 2)  # Replace this with your actual data
array2 = np.random.rand(214, 2)   # Replace this with your actual data

# Expand dimensions of array1 to have shape (1039, 1, 2)
expanded_array1 = array1[:, np.newaxis, :]
print(expanded_array1.shape)
# Create a 3D array by broadcasting array1 and array2
result = expanded_array1 + array2
print(result.shape)
# The result will be a 3D array with shape (1039, 214, 2)