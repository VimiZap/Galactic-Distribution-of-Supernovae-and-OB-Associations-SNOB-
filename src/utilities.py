import numpy as np
r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used

def rho_func(l, b, r):
    """
    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        distance from the Galactic center to the star/ a point in the Galaxy
    """
    return np.sqrt((r * np.cos(b))**2 + r_s**2 - 2 * r_s * r * np.cos(b) * np.cos(l)) # kpc, distance from the Sun to the star/ spacepoint


def theta_func(l, b, r):
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