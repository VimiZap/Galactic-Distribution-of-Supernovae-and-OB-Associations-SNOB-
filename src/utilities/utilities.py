import numpy as np
import time
import logging 
r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center


def rho(r, l, b):
    """ Function to calculate the distance from the Galactic center to the star/ a point in the Galaxy from Earth-centered coordinates (r, l, b)

    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        distance from the Galactic center to the star/ a point in the Galaxy
    """
    return np.sqrt((r * np.cos(b))**2 + r_s**2 - 2 * r_s * r * np.cos(b) * np.cos(l)) # kpc, distance from the Sun to the star/ spacepoint


def theta(r, l, b):
    """ Function to calculate the angle from the Galactic centre to the star/ a point in the Galaxy from Earth-centered coordinates (r, l, b)

    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        angle from the Sun to the star/ a point in the Galaxy
    """
    return np.arctan2(r_s - r*np.cos(b)*np.cos(l), r * np.cos(b) * np.sin(l))


def z(r, b):
    """ Function to calculate the z-coordinate of the star/ a point in the Galaxy from Earth-centered coordinates (r, l, b)

    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        b: Galactic latitude, radians
    Returns:
        z-coordinate of the star/ a point in the Galaxy
    """
    return r * np.sin(b)


def axisymmetric_disk_population(rho, h):
    """ Function describing the density of the disk at a distance rho from the Galactic center

    Args:
        rho: distance from the Galactic center
        h: scale length of the disk
    Returns:
        density of the disk at a distance rho from the Galactic center
    """
    return np.exp(-rho/h)


def height_distribution(z, sigma):
    """ Function describing the density of the disk at a height z above the Galactic plane

    Args:
        z: height above the Galactic plane
    Returns:
        height distribution of the disk at a height z above the Galactic plane
    """
    return np.exp(-0.5 * z**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def running_average(data, window_size):
   """ Calculates the running average of the data

   Args: 
        data: 1D np.array with data
        window_size: int, the size of the window used to calculate the running average. Denotes the number of points for each window
    Returns:
        1D np.array with the running average of the data
   """
   array_running_averaged = []
   delta = int((window_size)//2)
   print("running average: ", window_size, delta)
   for i in range(len(data)):
      if i-delta < 0:
         val = np.sum(data[-delta + i:]) + np.sum(data[:delta + i + 1])
         array_running_averaged.append(val)
      elif i+delta >= len(data):
         val = np.sum(data[i-delta:]) + np.sum(data[:delta + i - len(data) + 1])
         array_running_averaged.append(val)
      else:
         array_running_averaged.append(np.sum(data[i-delta:i+delta + 1]))
   return np.array(array_running_averaged)


def sum_pairwise(data):
    """ Sums up the elements of an array pairwise. Array must contain even number of points

    Args:
        a: even 1D np.array
    Returns:
        1D np.array with the summed up values. Half the size of the input array
    """
    if not len(data) % 2 == 0:
        print("The array must contain an even number of points")
        return None
    paired_data = data.reshape(-1, 2)
    result = np.sum(paired_data, axis=1)  # Sum along the specified axis (axis=1 sums up each row)
    return result


def rearange_data(data):
    """ Rearanges data to be plotted in desired format. E.g. instead of data going from 0 to 360 degrees, the returned data will go from 180 -> 0/ 360 -> 180 degrees, the format used by FIXEN et al. 1999 and Higdon and Lingenfelter 2013
    
    Args:
        data: 1D np.array with data. Must contain even number of points
    Returns:
        1D np.array with the rearanged data. Half the size of the input array
    """
    if not len(data) % 2 == 0:
        print("The array must contain an even number of points")
        return None
    middle = int(len(data)/2)
    data_centre_left = data[0]
    data_left = sum_pairwise(data[1:middle-1])
    data_left_edge = data[middle-1]
    data_right_edge = data[middle]
    data_edge = (data_right_edge + data_left_edge)
    data_right = sum_pairwise(data[middle+1:-1])
    data_centre_right = data[-1]
    data_centre = (data_centre_left + data_centre_right)
    rearanged_data = np.concatenate(([data_edge], data_left[::-1], [data_centre], data_right[::-1], [data_edge]))
    return rearanged_data


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"{func.__name__} took {elapsed_time:.6f} seconds to run.")
        return result
    return wrapper

