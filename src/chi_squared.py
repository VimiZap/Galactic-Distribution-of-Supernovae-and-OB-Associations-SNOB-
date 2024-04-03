import numpy as np
import matplotlib.pyplot as plt
import src.observational_data.firas_data as firas_data
import src.utilities.constants as const
import src.utilities.settings as settings
import src.spiral_arm_model as sam
from scipy import stats
from scipy.interpolate import griddata
import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)




def chi_squared(observational_data, observational_data_variance, modelled_data):
    # Ensure that we have no zeros in the modelled data set - set any zero equal to the minimal value in the array as to avoid dividing by zero
    modelled_data[modelled_data == 0] = np.min(modelled_data[modelled_data > 0])
    chi_squared = np.sum(((observational_data - modelled_data) ** 2) / observational_data_variance)
    return chi_squared


def load_firas_data():
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    firas_variance = (line_flux_error / 2) ** 2
    intensities_modelled = load_modelled_data()
    # the length of the firas data and modelled data differ. We need to expand the firas data to match the modelled data
    longitudes = np.linspace(0, 360, len(intensities_modelled))
    binned_longitudes, bin_edges = np.histogram(np.linspace(0, 360, len(longitudes)), bin_edges_line_flux)
    expanded_firas_intensity = np.repeat(line_flux, binned_longitudes)
    expanded_firas_variance = np.repeat(firas_variance, binned_longitudes)
    return expanded_firas_intensity, expanded_firas_variance


def load_modelled_data(filename_arm_intensities='intensities_per_arm_b_max_5.npy'):
    """ Function to load all components of the total NII intensity
    """
    intensities_per_arm = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/{filename_arm_intensities}')
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    intensities_modelled = np.sum(intensities_per_arm[:4], axis=0)
    if settings.add_local_arm_to_intensity_plot == True: # add the local arm contribution
        intensities_modelled += intensities_per_arm[4]
    if settings.add_devoid_region_sagittarius == True: # take into account the known devoid region of Sagittarius
        intensities_modelled += intensities_per_arm[5]
    if settings.add_gum_cygnus == True: # add the contribution from the nearby OBA
        gum = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_gum.npy')
        cygnus = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_cygnus.npy')
        gum_cygnus = gum + cygnus
        intensities_modelled += gum_cygnus
    return intensities_modelled


def interpolate_density_one_arm(h, arm_angle, pitch_angle, transverse_distances, transverse_densities_initial, arm_index):
    """ Integrates the densities of a single spiral arm over the entire galactic plane. The returned density is in units of kpc^-2. 
    Compared with the paper, it integrates P_\rho x P_\Delta at the top of page 6

    Args:
        grid_x (2D np.array): Contains all the x-values for the grid
        grid_y (2D np.array): Contaqins all the y-values for the grid
        method (str, optional): Interpolation method used in scipys griddata. Defaults to 'linear'.

    Returns:
        3D np.array: Interpolated densities for each spiral arm along axis 0. Axis 1 and 2 are the densities with respect to the grid
    """
    
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy')
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy')
    num_grid_subdivisions = settings.num_grid_subdivisions
    if num_grid_subdivisions < 1:
        raise ValueError("num_grid_subdivisions must be larger than 0")
    # generate the spiral arm medians
    theta, rho = sam.spiral_arm_medians(arm_angle, pitch_angle, const.rho_min_spiral_arm[arm_index], const.rho_max_spiral_arm[arm_index])
    # generate the spiral arm points
    x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, pitch_angle)
    # generate the spiral arm densities
    density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h)
    for sub_grid in range(num_grid_subdivisions):
        if sub_grid == num_grid_subdivisions - 1:
            x_grid_sub = x_grid[sub_grid * int(len(x_grid) / num_grid_subdivisions):]
            y_grid_sub = y_grid[sub_grid * int(len(y_grid) / num_grid_subdivisions):]
        else:
            x_grid_sub = x_grid[sub_grid * int(len(x_grid) / num_grid_subdivisions): (sub_grid + 1) * int(len(x_grid) / num_grid_subdivisions)]
            y_grid_sub = y_grid[sub_grid * int(len(y_grid) / num_grid_subdivisions): (sub_grid + 1) * int(len(y_grid) / num_grid_subdivisions)]
        # calculate interpolated density for the spiral arm
        interpolated_arm = griddata((x, y), density_spiral_arm, (x_grid_sub, y_grid_sub), method='cubic', fill_value=0)
        interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
        np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{arm_index}_{sub_grid}.npy', interpolated_arm)
    return      


def optimize_spiral_arm_start_angle(delta):
    """
    Args:
        delta: float, the step size for the arm angles to be tested. Can either be an integer, or a float between 0 and 1
    """
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Get the function name
    function_name = os.path.splitext(os.path.basename(__file__))[0]
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"{function_name}_{current_time}.txt")
    ###
    firas_intensity, firas_variance = load_firas_data()
    arm_angles = const.arm_angles.copy() - delta - 1 # subtract delta from each arm. Subtract 1 to make sure the first iteration is correct
    best_angles = const.arm_angles.copy()
    # check if delta is an integer or a float
    if delta == 0:
        logging.warning("delta is equal to zero. No optimization will be performed. Exiting...")
        return
    elif delta >= 1:
        scale = 1
    else: 
        scale = 10
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(const.sigma_arm) 
    num_angles_to_sample = delta * scale * 2 + 1 # multiply by 2 and add 1 to sample angles in range existing_angles +- delta.
    chi_squared_min = np.inf
    with open(filename_log, 'w') as file:
        # i = Norma-Cygnus, j = Perseus, k = Sagittarius-Carina, l = Scutum-Crux
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Chi-squared\n')
        for i in range(num_angles_to_sample): 
            arm_angles[0] += 1 / scale
            interpolate_density_one_arm(const.h_axisymmetric, arm_angles[0], const.pitch_angles[0], transverse_distances, transverse_densities_initial, arm_index=0)
            for j in range(num_angles_to_sample): 
                arm_angles[1] += 1 / scale
                interpolate_density_one_arm(const.h_axisymmetric, arm_angles[1], const.pitch_angles[1], transverse_distances, transverse_densities_initial, arm_index=1)
                for k in range(num_angles_to_sample):
                    arm_angles[2] += 1 / scale
                    interpolate_density_one_arm(const.h_axisymmetric, arm_angles[2], const.pitch_angles[2], transverse_distances, transverse_densities_initial, arm_index=2)
                    for l in range(num_angles_to_sample): 
                        arm_angles[3] += 1 / scale
                        interpolate_density_one_arm(const.h_axisymmetric, arm_angles[3], const.pitch_angles[3], transverse_distances, transverse_densities_initial, arm_index=3)
                        intensities_modelled = load_modelled_data()
                        chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                        file.write(f'{arm_angles[0]}, {arm_angles[1]}, {arm_angles[2]}, {arm_angles[3]}, {chi_squared_val}\n')
                        if chi_squared_val < chi_squared_min:
                            chi_squared_min = chi_squared_val
                            best_angles = arm_angles.copy()
    print('Best angles:', best_angles)
    print('Best chi-squared:', chi_squared_min)




def temp():
    firas_intensity, firas_variance = load_firas_data()
    intensities_modelled = load_modelled_data()

    chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
    print('calculated chi-squared:', chi_squared_val)



def main():
    optimize_spiral_arm_start_angle(delta=1)


if __name__ == '__main__':
    main()

