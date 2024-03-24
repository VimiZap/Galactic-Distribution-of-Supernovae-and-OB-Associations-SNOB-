import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import src.utilities.constants as const
import src.utilities.utilities as ut
import matplotlib.pyplot as plt
import src.spiral_arm_model as sam
import src.observational_data.analysis_obs_data as aod
import src.utilities.settings as settings
from scipy.interpolate import griddata





def add_local_arm_to_ax(ax):
    """ Add the local arm to the plot
    
    Args:
        ax: axis. The axis to add the spiral arms to
    
    Returns:
        None
    """
    theta, rho = sam.spiral_arm_medians(const.theta_start_local, const.pitch_local, const.rho_min_local, const.rho_max_local)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    ax.plot(x, y, color='black', marker='o', linewidth = 0.0001, zorder=6, markeredgewidth=0.0001, markersize=0.0001)
    return


def plot_arm_structure():
    fig, ax = plt.subplots(figsize=(10, 6))
    aod.add_spiral_arms_to_ax(ax)
    add_local_arm_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', s=15, zorder=11)
    ax.text(-1, 0.5, 'Galactic centre', fontsize=8, zorder=7)
    plt.title('Distribution of known and modelled associations in the Galactic plane')
    plt.xlabel('x (kpc)')
    plt.ylabel('y (kpc)')
    plt.xlim(-7.5, 7.5)
    plt.ylim(-2, 12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True, zorder=-10)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/local_arm_test.pdf')
    plt.close()


def interpolate_local_arm_density(h=const.h_spiral_arm, sigma_arm=const.sigma_arm):
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy')
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy')
    # generate the local arm medians
    theta, rho = sam.spiral_arm_medians(const.theta_start_local, const.pitch_local, const.rho_min_local, const.rho_max_local)
    # generate the spiral arm points
    x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, const.pitch_local)
    # generate the spiral arm densities
    density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h)
    # calculate interpolated density for the spiral arm
    num_grid_subdivisions = settings.num_grid_subdivisions
    if num_grid_subdivisions < 1:
        raise ValueError('num_grid_subdivisions has to be at least 1')
    # interpolate the local arm over each grid subdivision
    for sub_grid in range(num_grid_subdivisions):
        if sub_grid == num_grid_subdivisions - 1:
            x_grid_sub = x_grid[sub_grid * int(len(x_grid) / num_grid_subdivisions):]
            y_grid_sub = y_grid[sub_grid * int(len(y_grid) / num_grid_subdivisions):]
        else:
            x_grid_sub = x_grid[sub_grid * int(len(x_grid) / num_grid_subdivisions): (sub_grid + 1) * int(len(x_grid) / num_grid_subdivisions)]
            y_grid_sub = y_grid[sub_grid * int(len(y_grid) / num_grid_subdivisions): (sub_grid + 1) * int(len(y_grid) / num_grid_subdivisions)]
        # calculate interpolated density for the local arm arm
        interpolated_arm = griddata((x, y), density_spiral_arm, (x_grid_sub, y_grid_sub), method='cubic', fill_value=0)
        interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
        np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{5}_{sub_grid}.npy', interpolated_arm)
    return


def calc_modelled_emissivity_local_arm(b_max=1, db_above_1_deg = 0.1, fractional_contribution=const.fractional_contribution, readfile=True, h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles):
    logging.info("Calculating modelled emissivity of the Milky Way")
    if readfile == True:
        effective_area = np.load(f'{const.FOLDER_GALAXY_DATA}/effective_area_per_spiral_arm.npy')
    elif readfile == False:
        effective_area = sam.calc_effective_area_per_spiral_arm(h, sigma_arm, arm_angles, pitch_angles)
    else:
        raise ValueError("readfile must be either True or False")
    sam.calculate_galactic_coordinates(b_max, db_above_1_deg)
    logging.info("Coordinates calculated. Now interpolating each spiral arm")
    # coordinates made. Now we need to interpolate each spiral arm and sum up the densities
    interpolate_local_arm_density(h, sigma_arm)
    logging.info("Interpolation done. Now calculating the emissivity")
    common_multiplication_factor = const.total_galactic_n_luminosity * np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/height_distribution_values.npy')
    for i in range(4): # loop trough the 4 spiral arms
        logging.info(f"Calculating emissivity for spiral arm number: {i+1}")
        scaled_arm_emissivity = np.load(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{i}_0.npy') 
        for j in range(1, settings.num_grid_subdivisions): # loop through the different grid subdivisions
            scaled_arm_emissivity = np.concatenate((scaled_arm_emissivity, np.load(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{i}_{j}.npy')))
        scaled_arm_emissivity *= common_multiplication_factor * fractional_contribution[i] / (effective_area[i] * const.kpc**2) # multiply with the factors that makes the emissivity in units of erg/s/cm^2/sr
        # save the emissivity for the arm to disk
        np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_emissivity_{i}.npy', scaled_arm_emissivity)
    return  


def calc_modelled_intensity_local_arm(b_max=5, db_above_1_deg = 0.2, fractional_contribution=const.fractional_contribution, readfile=True, h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles):
    logging.info("Calculating the modelled NII intensity of the Milky Way")
    calc_modelled_emissivity_local_arm(b_max, db_above_1_deg, fractional_contribution, readfile, h, sigma_arm, arm_angles, pitch_angles)
    num_rads = len(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy'))
    num_lats = len(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/latitudes.npy'))
    dr = np.load(f'{const.FOLDER_GALAXY_DATA}/dr.npy')
    dl = np.load(f'{const.FOLDER_GALAXY_DATA}/dl.npy')
    db = np.load(f'{const.FOLDER_GALAXY_DATA}/db.npy')
    
    latitudinal_cosinus = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/latitudinal_cosinus.npy')
    common_multiplication_factor =  dr * latitudinal_cosinus/ (4 * np.pi * np.radians(b_max * 2) * np.radians(5))
    common_multiplication_factor = np.reshape(common_multiplication_factor, (num_rads, num_longs, num_lats)) * db[np.newaxis, np.newaxis, :] # reshaping to facilitate the multiplication with non-uniform latitudinal increments db
    common_multiplication_factor = common_multiplication_factor.ravel() #unraveling so that we can multiply with the interpolated densities
    del latitudinal_cosinus
    gc.collect()
    intensities_per_arm = np.zeros((4, num_longs)) # to store the intensity as a function of longitude for each spiral arm. Used for the intesnity-plots to compare with Higdon & Lingenfelter
    for i in range(4):
        logging.info(f"Calculating intensity for spiral arm number: {i+1}")
        arm_intensity = np.load(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_emissivity_{i}.npy') * common_multiplication_factor # spiral arms
        # reshape this 1D array into 3D array to facilitate for the summation over the different longitudes
        arm_intensity = arm_intensity.reshape((num_rads, num_longs, num_lats))
        # sum up to get the intensity as a function of longitude
        arm_intensity = arm_intensity.sum(axis=(0, 2)) # sum up all densities for all LOS for each value of longitude
        window_size = 5 / np.degrees(dl) # 5 degrees in divided by the increment in degrees for the longitude. This is the window size for the running average, number of points
        arm_intensity = ut.running_average(arm_intensity, window_size) /window_size # running average to smooth out the density distribution
        intensities_per_arm[i] += arm_intensity
    b_filename = str(b_max).replace(".", "_")
    filename_intensity_data = f'{const.FOLDER_GALAXY_DATA}/intensities_per_arm_b_max_{b_filename}.npy'
    np.save(filename_intensity_data, intensities_per_arm) # saving the same array we are plotting usually. Sum over all spiral arms to get one longitudinal map. With running average
    return


def main():
    #plot_arm_structure()
    interpolate_local_arm_density()

if __name__ == '__main__':
    main()