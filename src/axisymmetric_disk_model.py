import numpy as np
import matplotlib.pyplot as plt
import utilities.utilities as ut
import logging
logging.basicConfig(level=logging.INFO)
import observational_data.firas_data as firas_data
import src.utilities.constants as const


def calc_modelled_intensity(b_max = 5):
    logging.info("Running calc_modelled_intensity() for axisymmetric model")
    logging.info("Calculating coordinates")
    # Calculate coordinates    
    dr = 0.01   # increments in dr (kpc). For the spiral arm model, 0.01 kpc was used, but seems like 0.1 kpc is enough for the axisymmetric model
    dl = 0.2   # increments in dl (degrees)
    db = 0.1   # increments in db (degrees)
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(dr, const.r_s + const.rho_max_axisymmetric + dr, dr) #r_s + rho_max is the maximum distance from the Sun to the outer edge of the Galaxy
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    dl = np.radians(dl)
    # np.array with values for galactic latitude b in radians.
    latitudes = np.radians(np.arange(-b_max, b_max + db, db))
    db = np.radians(db)
    num_rads, num_longs, num_lats = len(radial_distances), len(longitudes), len(latitudes)
    # Create meshgrids. Effciciently creates 3D arrays with the coordinates for r, l and b
    radial_grid, long_grid, lat_grid = np.meshgrid(radial_distances, longitudes, latitudes, indexing='ij')
    latitudinal_cosinus = np.cos(lat_grid.ravel())
    height_distribution_values = ut.height_distribution(ut.z(radial_grid.ravel(), lat_grid.ravel()), sigma=const.sigma_height_distr)
    # Calculate the common multiplication factor for the modelled intensity. This factor ensures that, after the summation of the relative_density array further down, the result is the integrated intensity
    common_multiplication_factor =  dr * db * latitudinal_cosinus * height_distribution_values/ (4 * np.pi * np.radians(b_max * 2) * np.radians(5) * const.a_d_axisymmetric * const.kpc**2) 
    rho = ut.rho(radial_grid.ravel(), long_grid.ravel(), lat_grid.ravel())
    logging.info("Coordinates calculated. Calculating modelled intensity")
    relative_density = ut.axisymmetric_disk_population(rho, const.h_axisymmetric) * common_multiplication_factor # 3D array with the modelled relative density for each value of r, l and b
    logging.info("Modelled intensity calculated. Removing values outside the bright H 2 regions and normalizing to the measured value at 30 degrees longitude")
    mask = (rho > const.rho_max_axisymmetric) | (rho < const.rho_min_axisymmetric)
    # Set values in intensities to zero where the mask is True
    relative_density[mask] = 0
    intensities = np.sum(relative_density.reshape(num_rads, num_longs, num_lats), axis=(0, 2)) # sum over the radial and latitudinal axis
    # Now normalize the modelled intensity to the measured value at 30 degrees longitude in the FIRAS data
    abs_diff = np.abs(longitudes - np.radians(30))  # Calculate the absolute differences between the 30 degrees longitude and all elements in longitudes
    closest_index = np.argmin(abs_diff) # Find the index of the element with the smallest absolute difference
    modelled_value_30_degrees = intensities[closest_index] # Retrieve the closest value from the integrated spectrum
    luminosity_axisymmetric = const.measured_nii_30_deg / modelled_value_30_degrees # Calculate the normalization factor
    intensities *= luminosity_axisymmetric # normalize the modelled emissivity to the measured value at 30 degrees longitude    
    # Note: for comparison reasons with Higdon and Lingenfelter I am not using a running average on the intensities, as it smoothen the spikes
    logging.info("Saving the modelled intensity and the normalization factor")
    np.save(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_luminosity.npy', luminosity_axisymmetric)
    np.save(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_intensities.npy', intensities) 
    np.save(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_longitudes.npy', longitudes) 
    return


def plot_axisymmetric():
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_longitudes.npy')
    intensities = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_intensities.npy')
    luminosity = np.load(f'{const.FOLDER_GALAXY_DATA}/axisymmetric_luminosity.npy')
    fig, ax = plt.subplots()
    firas_data.add_firas_data_to_plot(ax)
    ax.plot(np.linspace(0, 360, len(longitudes)), intensities)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    ax.set_xticks(np.linspace(0, 360, 13))
    ax.set_xticklabels(x_ticks)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel("Galactic longitude l (degrees)")
    ax.set_ylabel("Modelled emissivity")
    ax.set_title("Modelled emissivity of the Galactic disk")
    ax.text(0.02, 0.95, fr'$H_\rho$ = {const.h_axisymmetric} kpc & $\sigma_z$ = {const.sigma_height_distr} kpc', transform=ax.transAxes, fontsize=8, color='black')
    ax.text(0.02, 0.9, fr'NII Luminosity = {luminosity:.2e} erg/s', transform=ax.transAxes, fontsize=8, color='black')
    ax.text(0.02, 0.85, fr'{const.rho_min_axisymmetric:.2e}  $\leq \rho \leq$ {const.rho_max_axisymmetric:.2e} kpc', transform=ax.transAxes, fontsize=8, color='black')
    fig.savefig(f'{const.FOLDER_MODELS_GALAXY}/axisymmetric_modelled_emissivity_h_2.5.pdf')  # save plot in the output folder
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO) 
    calc_modelled_intensity()
    plot_axisymmetric()


if __name__ == "__main__":
    main()
