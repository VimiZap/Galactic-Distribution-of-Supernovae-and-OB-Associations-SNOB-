import numpy as np

def slice_numpy_array(array, start, end):
    """ Function to remove a specified slice from a numpy array based on start and end indices.

    Args:
        array (np.array): The array from which to remove the slice
        start (int): The start index of the slice to remove
        end (int): The end index of the slice to remove
    Returns:
        np.array: The array with the specified slice removed
    """
    if start > end:
        raise ValueError("The start index must be smaller than the end index")
    if start < 0 or end >= len(array):  # Adjust the condition to prevent out-of-bounds access
        raise ValueError("The start and end indices must be within the length of the array")

    # Use np.delete to remove elements from start to end (inclusive of end)
    return np.delete(array, slice(start, end + 1))

# Example usage
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(slice_numpy_array(arr, 3, 5))  # Expected output: [0, 1, 2, 6, 7, 8, 9]
