import numpy as np
import logging
import src.utilities.constants as const
import src.utilities.utilities as ut

def gaussian_distribution(x, sigma):
    return np.exp(-0.5 * x**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def generate_uniform_sphere(radius):
    # Create a 3D grid
    dr = radius / 50
    x = np.arange(-radius, radius + dr, dr)
    y = np.arange(-radius, radius + dr, dr)
    z = np.arange(-radius, radius + dr, dr)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    # Calculate distance from the origin for each point
    distance = np.sqrt(x**2 + y**2 + z**2)
    # Mask points inside the sphere
    mask = distance <= radius
    x = x[mask]
    y = y[mask]
    z = z[mask]
    fwhm = np.radians(7)
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # Assign density values to each point
    density_values = gaussian_distribution(distance[mask], std) #np.ones_like(distance[mask]) #np.exp(-distance[mask]**2)  # Example: Gaussian density function
    #density_values *= 1 / gaussian_distribution(0, std)  # Normalize density values
    return x, y, z, density_values


def sum_z_values(x, y, density_values):
    # Sum up z-values for each column in the xy plane
    unique_xy_pairs = np.unique(np.column_stack((x, y)), axis=0)
    summed_z_values = np.zeros_like(unique_xy_pairs[:, 0])

    for i, xy_pair in enumerate(unique_xy_pairs):
        mask = (x == xy_pair[0]) & (y == xy_pair[1])
        summed_z_values[i] = np.sum(density_values[mask])

    return unique_xy_pairs[:, 0], unique_xy_pairs[:, 1], summed_z_values


def generate_gum_cygnus():
    # Cygnus parameters
    cygnus_distance = 1.45 # kpc
    cygnus_long = np.radians(80)
    cygnus_lat = np.radians(0)
    cygnus_radius = 0.075 # kpc
    cygnus_rho = ut.rho(cygnus_distance, cygnus_long, cygnus_lat)
    cygnus_theta = ut.theta(cygnus_distance, cygnus_long, cygnus_lat)
    cygnus_x = cygnus_rho * np.cos(cygnus_theta) 
    cygnus_y = cygnus_rho * np.sin(cygnus_theta)
    # Gum parameters
    gum_distance = 0.33 # kpc
    gum_long = np.radians(262)
    gum_lat = np.radians(0)
    gum_radius = 0.03 # kpc
    gum_rho = ut.rho(gum_distance, gum_long, gum_lat)
    gum_theta = ut.theta(gum_distance, gum_long, gum_lat)
    gum_x = gum_rho * np.cos(gum_theta)
    gum_y = gum_rho * np.sin(gum_theta)
    # Generate spheres
    c_x, c_y, c_z, c_density_values = generate_uniform_sphere(cygnus_radius)
    c_x += cygnus_x
    c_y += cygnus_y
    g_x, g_y, g_z, g_density_values = generate_uniform_sphere(gum_radius)
    g_x += gum_x
    g_y += gum_y
    # Sum up z-values for each column in the xy plane
    summed_c_x, summed_c_y, densities_c = sum_z_values(c_x, c_y, c_density_values)
    summed_g_x, summed_g_y, densities_g = sum_z_values(g_x, g_y, g_density_values)
    return summed_c_x, summed_c_y, densities_c, summed_g_x, summed_g_y, densities_g