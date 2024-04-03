import numpy as np
import logging
import src.utilities.constants as const
import src.utilities.utilities as ut
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import src.observational_data.firas_data as firas_data


def gaussian_distribution(l, mu, sigma):
    return np.exp(-0.5 * (l - mu)**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def generate_uniform_sphere(radius):
    # Create a 3D grid
    dr = radius / 180 # the smaller the value I am dividing by, the larger the value for the modelled intensity
    print('dr = ', dr)
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
    # Calculate the density values/ intensity
    density_values = np.ones(len(x)) * dr / (4 * np.pi * const.kpc**2 * np.radians(5) * np.radians(10))
    return x, y, z, density_values


def generate_uniform_sphere_2(radius, center_l, center_b, dr_factor=150):
    dr = radius / dr_factor
    r_values = np.arange(0, radius + dr, dr)
    
    # Assuming the sphere is small compared to the size of the galaxy,
    # so we can approximate within a small range of l and b
    l_values = np.linspace(center_l - radius, center_l + radius, dr_factor)
    b_values = np.linspace(center_b - radius, center_b + radius, dr_factor)
    
    # Create a grid in spherical coordinates
    r, l, b = np.meshgrid(r_values, l_values, b_values, indexing='ij')
    
    # Convert to Cartesian coordinates for the distance calculation
    x = r * np.cos(l) * np.cos(b)
    y = r * np.sin(l) * np.cos(b)
    z = r * np.sin(b)
    
    # Calculate distance from the center to each point
    distance = np.sqrt((x)**2 + (y)**2 + (z)**2)
    
    # Mask points inside the sphere
    mask = distance <= radius
    r = r[mask]
    l = l[mask]
    b = b[mask]
    
    ###
    x = r * np.cos(l) * np.cos(b)
    y = r * np.sin(l) * np.cos(b)
    z = r * np.sin(b)
    ###
    # Calculate the density values/intensity for each point
    # Adjust this as necessary for your model
    density_values = np.ones(r.shape) # * dr / (4 * np.pi * radius**2)
    return x, y, z, density_values
    return r, l, b, density_values


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
    """ # Sum up z-values for each column in the xy plane
    summed_c_x, summed_c_y, densities_c = sum_z_values(c_x, c_y, c_density_values)
    summed_g_x, summed_g_y, densities_g = sum_z_values(g_x, g_y, g_density_values)
    return summed_c_x, summed_c_y, densities_c, summed_g_x, summed_g_y, densities_g """
    # retrieve the longitudes
    c_longitudes = ut.xy_to_long(c_x, c_y)
    g_longitudes = ut.xy_to_long(g_x, g_y)
    # calculate the weights for the density values, based on the gaussian profile
    fwhm = np.radians(7)
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))
    c_density_weights = gaussian_distribution(c_longitudes, cygnus_long, std)
    g_density_weights = gaussian_distribution(g_longitudes, gum_long, std)
    c_densities = c_density_values * c_density_weights * const.cygnus_nii_luminosity
    g_densities = g_density_values * g_density_weights * const.gum_nii_luminosity
    return c_longitudes, c_densities, g_longitudes, g_densities


def test_3d_sphere():
    x, y, z, density_values = generate_uniform_sphere_2(0.075, 0, 0, dr_factor=10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=density_values, marker='o', label='Sphere')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal')
    plt.legend()
    plt.show()
    plt.close()


def test_sphere_to_long():
    x, y, z, density_values = generate_uniform_sphere(0.075)
    longitudes = ut.xy_to_long(x, y)
    """ plt.hist(longitudes, bins=100)
    plt.show()
    plt.close() """
    dl = 0.1
    bins = np.arange(0, 360 + dl, dl)
    c_longitudes, c_densities, g_longitudes, g_densities = generate_gum_cygnus()
    g_intensity, bin_edges = np.histogram(np.degrees(g_longitudes), bins=bins, weights=g_densities)
    #rearanged_bins = ut.rearange_data(bins[])
    print('len g_intensity: ', len(g_intensity))
    rearanged_g_intensity = ut.rearange_data(g_intensity)

    print('bin_edges generated by np.hist: ', bin_edges)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers = ut.rearange_data(bin_centers)
    #dl = bin_centers[1] - bin_centers[0]
    print('dl = ', dl)
    window_size = 5 / dl # 5 degrees in divided by the increment in degrees for the longitude. This is the window size for the running average, number of points
    rearanged_g_intensity = ut.running_average(rearanged_g_intensity, window_size) /window_size # running average to smooth out the density distribution
    print('len bin_centers: ', len(bin_centers))
    print('len rearanged_g_intensity: ', len(rearanged_g_intensity))
    plt.plot(np.linspace(0, 360, len(rearanged_g_intensity)), rearanged_g_intensity)
    #plt.hist(np.degrees(c_longitudes), bins=100, weights=c_densities, label='Cygnus')
    #plt.hist(np.degrees(g_longitudes), bins=100, weights=g_densities, label='Gum')
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()
    plt.show()


def plot_modelled_intensity_gum_cygnus(filename_output = f'{const.FOLDER_MODELS_GALAXY}/gym_cygnus_test.pdf'):
    dl = 0.1
    bins = np.arange(0, 360 + dl, dl)
    c_longitudes, c_densities, g_longitudes, g_densities = generate_gum_cygnus()
    g_intensity, bin_edges = np.histogram(np.degrees(g_longitudes), bins=bins, weights=g_densities)
    num_counts_g, _ = np.histogram(np.degrees(g_longitudes), bins=bins)
    print(f'shape g_intensity: {g_intensity.shape}, shape num_counts_g: {num_counts_g.shape}')
    rearanged_g_intensity = ut.rearange_data(g_intensity / num_counts_g)
    c_intensity, _ = np.histogram(np.degrees(c_longitudes), bins=bins, weights=c_densities)
    num_counts_c, _ = np.histogram(np.degrees(c_longitudes), bins=bins)
    rearanged_c_intensity = ut.rearange_data(c_intensity / num_counts_c)
    #bin_centers = (bins[:-1] + bins[1:]) / 2
    #bin_centers = ut.rearange_data(bin_centers)
    window_size = 5 / dl # 5 degrees in divided by the increment in degrees for the longitude. This is the window size for the running average, number of points
    print('dl, window_size: ', dl, window_size)
    rearanged_g_intensity = ut.running_average(rearanged_g_intensity, window_size) /window_size # running average to smooth out the density distribution
    rearanged_c_intensity = ut.running_average(rearanged_c_intensity, window_size) /window_size # running average to smooth out the density distribution
    #plt.plot(np.linspace(0, 360, len(rearanged_g_intensity)), rearanged_g_intensity)
    plt.plot(np.linspace(0, 360, len(rearanged_c_intensity)), rearanged_c_intensity)    
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()
    plt.show()


def asd_plot_modelled_intensity_gum_cygnus(filename_output = f'{const.FOLDER_MODELS_GALAXY}/gym_cygnus_test.pdf'):
    """ Plots the modelled intensity of the Galactic disk as a function of Galactic longitude. Each spiral arm is plotted separately, as well as the total intensity. Assumes that the itensities have been calculated and saved to disk beforehand.

    Args:
        filename_output (str, optional): Name of the output file. Defaults to f'{const.FOLDER_MODELS_GALAXY}/modelled_intensity.pdf'.
        filename_intensity_data (str, optional): Name of the file containing the intensity data. Defaults to f'{const.FOLDER_GALAXY_DATA}/intensities_per_arm.npy'.
        fractional_contribution (list, optional): Fractional contribution of each spiral arm. Defaults to const.fractional_contribution.
        h (float, optional): Scale length of the disk. Defaults to const.h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to const.sigma_arm.

    Returns:
        None: The plot is saved to disk
    """
    # plot the FIRAS data for the NII 205 micron line
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # plot the modelled intensity
    intensities_per_arm = None
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[0], label=f"")
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Line intensity in erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$")
    plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig(filename_output)
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO)
    #test_3d_sphere()
    #test_sphere_to_long()
    plot_modelled_intensity_gum_cygnus()
    

if __name__ == '__main__':
    main()

