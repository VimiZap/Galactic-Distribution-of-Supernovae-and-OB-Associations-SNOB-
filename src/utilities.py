import numpy as np
r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used

def rho(l, b, r):
    """
    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        distance from the Galactic center to the star/ a point in the Galaxy
    """
    return np.sqrt((r * np.cos(b))**2 + r_s**2 - 2 * r_s * r * np.cos(b) * np.cos(l)) # kpc, distance from the Sun to the star/ spacepoint


def theta(l, b, r):
    """
    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        angle from the Sun to the star/ a point in the Galaxy
    """
    return np.arctan2(r_s - r*np.cos(b)*np.cos(l), r * np.cos(b) * np.sin(l))


def z(r, b):
    """
    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        b: Galactic latitude, radians
    Returns:
        z-coordinate of the star/ a point in the Galaxy
    """
    return r * np.sin(b)


def height_distribution(z, sigma): # z is the height above the Galactic plane
    """
    Args:
        z: height above the Galactic plane
    Returns:
        height distribution of the disk at a height z above the Galactic plane
    """
    return np.exp(-0.5 * z**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def running_average(data, window_size):
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
