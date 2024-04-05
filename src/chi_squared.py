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
from pathlib import Path
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
        try: 
            intensities_modelled += intensities_per_arm[4]
        except: 
            logging.warning("The local arm has not been added to the modelled data. You can generate it in spiral_arm_model.py")
    if settings.add_devoid_region_sagittarius == True: # take into account the known devoid region of Sagittarius
        try:
            intensities_modelled += intensities_per_arm[5]
        except:
            logging.warning("The devoid region of Sagittarius has not been added to the modelled data. You can generate it in spiral_arm_model.py")
    if settings.add_gum_cygnus == True: # add the contribution from the nearby OBA
        try:
            gum = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_gum.npy')
            cygnus = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_cygnus.npy')
            gum_cygnus = gum + cygnus
            intensities_modelled += gum_cygnus
        except:
            logging.warning("The Gum Nebula and Cygnus Loop contributions have not been added to the modelled data. You can generate them in gum_cygnus.py")
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
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"start_angle_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    initial_arm_angles = np.degrees(const.arm_angles[:5].copy()) - delta # :5 to explicitly avoid the devoid region 
    arm_angles = np.degrees(const.arm_angles[:5].copy()) - delta # subtract delta from each arm. # :5 to explicitly avoid the devoid region 
    best_angles = np.degrees(const.arm_angles[:5].copy()) # :5 to explicitly avoid the devoid region 
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
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, Chi-squared\n')
        for i in range(num_angles_to_sample): 
            arm_angles[0] = initial_arm_angles[0] + i / scale
            interpolate_density_one_arm(const.h_spiral_arm, np.radians(arm_angles[0]), const.pitch_angles[0], transverse_distances, transverse_densities_initial, arm_index=0)
            for j in range(num_angles_to_sample): 
                arm_angles[1] = initial_arm_angles[1] + j / scale
                interpolate_density_one_arm(const.h_spiral_arm, np.radians(arm_angles[1]), const.pitch_angles[1], transverse_distances, transverse_densities_initial, arm_index=1)
                for k in range(num_angles_to_sample):
                    arm_angles[2] = initial_arm_angles[2] + k / scale
                    interpolate_density_one_arm(const.h_spiral_arm, np.radians(arm_angles[2]), const.pitch_angles[2], transverse_distances, transverse_densities_initial, arm_index=2)
                    for l in range(num_angles_to_sample): 
                        arm_angles[3] = initial_arm_angles[3] + l / scale
                        logging.info(f'Angles: {arm_angles[:4]}')
                        interpolate_density_one_arm(const.h_spiral_arm, np.radians(arm_angles[3]), const.pitch_angles[3], transverse_distances, transverse_densities_initial, arm_index=3)
                        sam.calc_modelled_intensity(readfile_effective_area=False, arm_angles=np.radians(arm_angles), interpolate_all_arms=False)
                        intensities_modelled = load_modelled_data()
                        chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                        file.write(f'{arm_angles[0]}, {arm_angles[1]}, {arm_angles[2]}, {arm_angles[3]}, {np.degrees(const.pitch_angles[0])}, {np.degrees(const.pitch_angles[1])}, {np.degrees(const.pitch_angles[2])}, {np.degrees(const.pitch_angles[3])}, {const.fractional_contribution[0]}, {const.fractional_contribution[1]}, {const.fractional_contribution[2]}, {const.fractional_contribution[3]}, {const.h_spiral_arm}, {const.sigma_arm}, {chi_squared_val}\n')
                        if chi_squared_val < chi_squared_min:
                            chi_squared_min = chi_squared_val
                            best_angles = arm_angles.copy()
    print('Best arm start angles:', best_angles)
    print('Best chi-squared:', chi_squared_min)
    return np.radians(best_angles)


def optimize_spiral_arm_pitch_angle(delta, best_spiral_arm_angles=const.arm_angles):
    """
    Args:
        delta: float, the step size for the arm angles to be tested. Can either be an integer, or a float between 0 and 1
    """
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"pitch_angle_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    initial_pitch_angles = np.degrees(const.pitch_angles[:4].copy()) - delta
    pitch_angles = np.degrees(const.pitch_angles[:4].copy()) - delta # subtract delta from each arm.
    best_pitch_angles = np.degrees(const.pitch_angles[:4].copy())
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
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, Chi-squared\n')
        for i in range(num_angles_to_sample): 
            pitch_angles[0] = initial_pitch_angles[0] + i / scale
            interpolate_density_one_arm(const.h_spiral_arm, best_spiral_arm_angles[0], np.radians(pitch_angles[0]), transverse_distances, transverse_densities_initial, arm_index=0)
            for j in range(num_angles_to_sample): 
                pitch_angles[1] = initial_pitch_angles[1] + j / scale
                interpolate_density_one_arm(const.h_spiral_arm, best_spiral_arm_angles[1], np.radians(pitch_angles[1]), transverse_distances, transverse_densities_initial, arm_index=1)
                for k in range(num_angles_to_sample):
                    pitch_angles[2] = initial_pitch_angles[2] + k / scale
                    interpolate_density_one_arm(const.h_spiral_arm, best_spiral_arm_angles[2], np.radians(pitch_angles[2]), transverse_distances, transverse_densities_initial, arm_index=2)
                    for l in range(num_angles_to_sample): 
                        pitch_angles[3] = initial_pitch_angles[3] + l / scale
                        logging.info(f'Pitch angles: {pitch_angles[:4]}')
                        interpolate_density_one_arm(const.h_spiral_arm, best_spiral_arm_angles[3], np.radians(pitch_angles[3]), transverse_distances, transverse_densities_initial, arm_index=3)
                        sam.calc_modelled_intensity(readfile_effective_area=False, arm_angles=best_spiral_arm_angles, pitch_angles=np.radians(pitch_angles), interpolate_all_arms=False)
                        intensities_modelled = load_modelled_data()
                        chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                        file.write(f'{np.degrees(best_spiral_arm_angles[0])}, {np.degrees(best_spiral_arm_angles[1])}, {np.degrees(best_spiral_arm_angles[2])}, {np.degrees(best_spiral_arm_angles[3])}, {pitch_angles[0]}, {pitch_angles[1]}, {pitch_angles[2]}, {pitch_angles[3]}, {const.fractional_contribution[0]}, {const.fractional_contribution[1]}, {const.fractional_contribution[2]}, {const.fractional_contribution[3]}, {const.h_spiral_arm}, {const.sigma_arm}, {chi_squared_val}\n')
                        if chi_squared_val < chi_squared_min:
                            chi_squared_min = chi_squared_val
                            best_pitch_angles = pitch_angles.copy()
    print('Best pitch angles:', best_pitch_angles)
    print('Best chi-squared:', chi_squared_min)
    return np.radians(best_pitch_angles)


def optimize_sigma_arm(best_spiral_arm_angles = const.arm_angles, best_pitch_angles=const.pitch_angles):
    """
    Args:
        delta: float, the step size for the arm angles to be tested. Can either be an integer, or a float between 0 and 1
    """
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"sigma_arm_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    
    best_sigma = const.sigma_arm.copy()
    sigmas_to_check = np.arange(0.25, 0.75 + 0.01, 0.01)
    chi_squared_min = np.inf
    with open(filename_log, 'w') as file:
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, Chi-squared\n')
        for sigma in sigmas_to_check:
            sam.calc_modelled_intensity(readfile_effective_area=False, sigma_arm=sigma, arm_angles=best_spiral_arm_angles, pitch_angles=best_pitch_angles, interpolate_all_arms=True)
            intensities_modelled = load_modelled_data()
            chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
            file.write(f'{np.degrees(best_spiral_arm_angles[0])}, {np.degrees(best_spiral_arm_angles[1])}, {np.degrees(best_spiral_arm_angles[2])}, {np.degrees(best_spiral_arm_angles[3])}, {np.degrees(best_pitch_angles[0])}, {np.degrees(best_pitch_angles[1])}, {np.degrees(best_pitch_angles[2])}, {np.degrees(best_pitch_angles[3])}, {const.fractional_contribution[0]}, {const.fractional_contribution[1]}, {const.fractional_contribution[2]}, {const.fractional_contribution[3]}, {const.h_spiral_arm}, {sigma}, {chi_squared_val}\n')
            if chi_squared_val < chi_squared_min:
                chi_squared_min = chi_squared_val
                best_sigma = sigma.copy()
    print('Best sigma_arm:', best_sigma)
    print('Best chi-squared:', chi_squared_min)
    return best_sigma


def optimize_spiral_arm_h(best_sigma_arm = const.sigma_arm, best_spiral_arm_angles = const.arm_angles, best_pitch_angles=const.pitch_angles):
    """
    Args:
        delta: float, the step size for the arm angles to be tested. Can either be an integer, or a float between 0 and 1
    """
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"h_spiral_arm_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    best_h_spiral_arm = const.h_spiral_arm.copy()
    h_spiral_arm_to_check = np.arange(1.5, 3.5 + 0.1, 0.1)
    chi_squared_min = np.inf
    with open(filename_log, 'w') as file:
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, Chi-squared\n')
        for h_spiral_arm in h_spiral_arm_to_check:
            sam.calc_modelled_intensity(readfile_effective_area=False, h = h_spiral_arm, sigma_arm=best_sigma_arm, arm_angles=best_spiral_arm_angles, pitch_angles=best_pitch_angles, interpolate_all_arms=True)
            intensities_modelled = load_modelled_data()
            chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
            file.write(f'{np.degrees(best_spiral_arm_angles[0])}, {np.degrees(best_spiral_arm_angles[1])}, {np.degrees(best_spiral_arm_angles[2])}, {np.degrees(best_spiral_arm_angles[3])}, {np.degrees(best_pitch_angles[0])}, {np.degrees(best_pitch_angles[1])}, {np.degrees(best_pitch_angles[2])}, {np.degrees(best_pitch_angles[3])}, {const.fractional_contribution[0]}, {const.fractional_contribution[1]}, {const.fractional_contribution[2]}, {const.fractional_contribution[3]}, {h_spiral_arm}, {best_sigma_arm}, {chi_squared_val}\n')
            if chi_squared_val < chi_squared_min:
                chi_squared_min = chi_squared_val
                best_h_spiral_arm = h_spiral_arm.copy()
    print('Best h_spiral_arm:', best_h_spiral_arm)
    print('Best chi-squared:', chi_squared_min)
    return best_h_spiral_arm


def optimize_fractional_contribution_four_spiral_arms(fractional_contribution, best_sigma_arm = const.sigma_arm, best_h_spiral_arm=const.h_spiral_arm, best_spiral_arm_angles = const.arm_angles, best_pitch_angles=const.pitch_angles):
    """
    Args:
        fractional_contribution: list of floats, the fractional contribution of each spiral arm. Should be four floats that sum to 1"""
    if np.sum(fractional_contribution) != 1:
        logging.warning(f"The fractional contributions do not sum to 1. The sum is {np.sum(fractional_contribution)}. Exiting...")
        return
    if len(fractional_contribution) != 4:
        logging.warning(f"The fractional contributions should be a list of four floats. The array contains {len(fractional_contribution)} elements. Exiting...")
        return
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename_log = os.path.join(const.FOLDER_CHI_SQUARED, f"fractional_four_spiral_arms_{current_time}.txt")
    # check if the folder exists, if not create it
    Path(const.FOLDER_CHI_SQUARED).mkdir(parents=True, exist_ok=True)
    firas_intensity, firas_variance = load_firas_data()
    best_fractional_contribution = fractional_contribution.copy()
    fractional_contributions_to_check = fractional_contribution.copy()
    sam.calc_modelled_intensity(fractional_contribution=fractional_contributions_to_check, interpolate_all_arms=False) # only checking fractional contribution - nothing to interpolate as fractional contribution is just some scale factor
    intensities_modelled = load_modelled_data()
    chi_squared_min = chi_squared(firas_intensity, firas_variance, intensities_modelled)
    increments = [+0.01, -0.01]
    with open(filename_log, 'w') as file:
        file.write('Arm_angle_NC, Arm_angle_P, Arm_angle_SA, Arm_angle_SC, Pitch_angle_NC, Pitch_angle_P, Pitch_angle_SA, Pitch_angle_SC, f_NC, f_P, f_SA, f_SC, h, sigma, Chi-squared\n')
        for i in range(len(fractional_contributions_to_check) - 1):
            for increment in increments:
                fractional_contributions_to_check[i] += increment
                for k in range(len(fractional_contributions_to_check[i+1:])):
                    fractional_contributions_to_check[i + 1 + k] -= increment
                    sam.calc_modelled_intensity(fractional_contribution=fractional_contributions_to_check, interpolate_all_arms=False)
                    intensities_modelled = load_modelled_data()
                    chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
                    file.write(f'{np.degrees(best_spiral_arm_angles[0])}, {np.degrees(best_spiral_arm_angles[1])}, {np.degrees(best_spiral_arm_angles[2])}, {np.degrees(best_spiral_arm_angles[3])}, {np.degrees(best_pitch_angles[0])}, {np.degrees(best_pitch_angles[1])}, {np.degrees(best_pitch_angles[2])}, {np.degrees(best_pitch_angles[3])}, {fractional_contributions_to_check[0]}, {fractional_contributions_to_check[1]}, {fractional_contributions_to_check[2]}, {fractional_contributions_to_check[3]}, {best_h_spiral_arm}, {best_sigma_arm}, {chi_squared_val}\n')
                    if chi_squared_val < chi_squared_min:
                        chi_squared_min = chi_squared_val
                        best_fractional_contribution = fractional_contributions_to_check.copy()
    print('Best fractional contribution:', best_fractional_contribution)
    print('Best chi-squared:', chi_squared_min)
    return best_fractional_contribution
            

def optimize_fractional_contribution_four_spiral_arms_total(fractional_contribution_original = const.fractional_contribution, best_sigma_arm = const.sigma_arm, best_h_spiral_arm=const.h_spiral_arm, best_spiral_arm_angles = const.arm_angles, best_pitch_angles=const.pitch_angles):
    fractional_contribution = fractional_contribution_original[:4].copy() # copy to prevent editing the original, and ensure we only pick out the first four elements which are the main spiral arms
    best_fractional_contribution = None
    if np.sum(fractional_contribution) != 1:
        logging.warning(f"The fractional contributions do not sum to 1. The sum is {np.sum(fractional_contribution)}. Exiting...")
        return
    if len(fractional_contribution) != 4:
        logging.warning(f"The fractional contributions should be a list of four floats. The array contains {len(fractional_contribution)} elements. Exiting...")
        return
    best_fractional_contribution = optimize_fractional_contribution_four_spiral_arms(fractional_contribution, best_sigma_arm, best_h_spiral_arm, best_spiral_arm_angles, best_pitch_angles)
    count = 0 # count the number of iterations
    while(best_fractional_contribution != fractional_contribution or count < 10):
        fractional_contribution = best_fractional_contribution.copy()
        best_fractional_contribution = optimize_fractional_contribution_four_spiral_arms(fractional_contribution, best_sigma_arm, best_h_spiral_arm, best_spiral_arm_angles, best_pitch_angles)
        count += 1
    print(f'The very best fractional contribution: {best_fractional_contribution}. Obtained after {count} iterations')
    return best_fractional_contribution
    

def run_tests(delta):
    """
    Function to run all the optimization tests
    """
    best_spiral_arm_angles = optimize_spiral_arm_start_angle(delta)
    best_pitch_angles = optimize_spiral_arm_pitch_angle(delta, best_spiral_arm_angles)
    best_sigma_arm = optimize_sigma_arm(best_spiral_arm_angles, best_pitch_angles)
    best_h_spiral_arm = optimize_spiral_arm_h(best_sigma_arm, best_spiral_arm_angles, best_pitch_angles)
    print('Best spiral arm angles:', best_spiral_arm_angles)
    print('Best pitch angles:', best_pitch_angles)
    print('Best sigma_arm:', best_sigma_arm)
    print('Best h_spiral_arm:', best_h_spiral_arm)

    #optimize_spiral_arm_fractional_contribution(delta)


def temp():
    firas_intensity, firas_variance = load_firas_data()
    intensities_modelled = load_modelled_data()

    chi_squared_val = chi_squared(firas_intensity, firas_variance, intensities_modelled)
    print('number of arms entering the calculation:', len(const.arm_angles))
    print('calculated chi-squared:', chi_squared_val)



def main():
    optimize_spiral_arm_start_angle(delta=1)


if __name__ == '__main__':
    #main()
    #temp()
    run_tests(delta=2)

