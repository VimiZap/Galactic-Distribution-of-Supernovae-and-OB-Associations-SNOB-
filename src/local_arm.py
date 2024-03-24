import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import src.utilities.constants as const
import src.utilities.utilities as ut
import matplotlib.pyplot as plt
import src.spiral_arm_model as sam
import src.observational_data.analysis_obs_data as aod
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


""" def interpolate_local_arm_density(h=const.h_spiral_arm, sigma_arm=const.sigma_arm):
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy')
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy')
    print('x_grid.shape', x_grid.shape, 'y_grid.shape', y_grid.shape)

    # generate the local arm medians
    theta, rho = sam.spiral_arm_medians(const.theta_start_local, const.pitch_local, const.rho_min_local, const.rho_max_local)
    # generate the spiral arm points
    x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, const.pitch_local)
    print('x.shape', x.shape, 'y.shape', y.shape)

    # generate the spiral arm densities
    density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h)
    # calculate interpolated density for the spiral arm
    interpolated_arm = griddata((x, y), density_spiral_arm, (x_grid, y_grid), method='cubic', fill_value=0)
    interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
    np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_local_arm.npy', interpolated_arm)
    return """


def interpolate_local_arm_density(h=const.h_spiral_arm, sigma_arm=const.sigma_arm):
    """ Integrates the densities of the spiral arms over the entire galactic plane. The returned density is in units of kpc^-2. 
    Compared with the paper, it integrates P_\rho x P_\Delta at the top of page 6

    Args:
        grid_x (2D np.array): Contains all the x-values for the grid
        grid_y (2D np.array): Contaqins all the y-values for the grid
        method (str, optional): Interpolation method used in scipys griddata. Defaults to 'linear'.

    Returns:
        3D np.array: Interpolated densities for each spiral arm along axis 0. Axis 1 and 2 are the densities with respect to the grid
    """
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy')
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy')
    print('x_grid.shape', x_grid.shape, 'y_grid.shape', y_grid.shape)
    print('datatype x_grid, y_grid', x_grid.dtype, y_grid.dtype)
    for i in range(len([const.theta_start_local])):
        # generate the spiral arm medians
        #theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i])
        theta, rho = sam.spiral_arm_medians(const.theta_start_local, const.pitch_local, const.rho_min_local, const.rho_max_local)

        # generate the spiral arm points
        x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, const.pitch_local)
        print('x.shape', x.shape, 'y.shape', y.shape)
        print('datatype x, y', x.dtype, y.dtype)
        # generate the spiral arm densities
        density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h)
        # calculate interpolated density for the spiral arm
        interpolated_arm = griddata((x, y), density_spiral_arm, (x_grid, y_grid), method='cubic', fill_value=0)
        interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
        np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_local_arm.npy', interpolated_arm)
    return



def interpolate_density(h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles):
    """ Integrates the densities of the spiral arms over the entire galactic plane. The returned density is in units of kpc^-2. 
    Compared with the paper, it integrates P_\rho x P_\Delta at the top of page 6

    Args:
        grid_x (2D np.array): Contains all the x-values for the grid
        grid_y (2D np.array): Contaqins all the y-values for the grid
        method (str, optional): Interpolation method used in scipys griddata. Defaults to 'linear'.

    Returns:
        3D np.array: Interpolated densities for each spiral arm along axis 0. Axis 1 and 2 are the densities with respect to the grid
    """
    transverse_distances, transverse_densities_initial = sam.generate_transverse_spacing_densities(sigma_arm) 
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy')
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy')
    print('x_grid.shape', x_grid.shape, 'y_grid.shape', y_grid.shape)
    print('datatype x_grid, y_grid', x_grid.dtype, y_grid.dtype)

    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(arm_angles[i], pitch_angles[i])
        # generate the spiral arm points
        x, y = sam.generate_spiral_arm_coordinates(rho, transverse_distances, theta, pitch_angles[i])
        print('x.shape', x.shape, 'y.shape', y.shape)
        print('datatype x, y', x.dtype, y.dtype)

        # generate the spiral arm densities
        density_spiral_arm = sam.generate_spiral_arm_densities(rho, transverse_densities_initial, h)
        # calculate interpolated density for the spiral arm
        interpolated_arm = griddata((x, y), density_spiral_arm, (x_grid, y_grid), method='cubic', fill_value=0)
        interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
        #np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{i}.npy', interpolated_arm)
    return


def main():
    #plot_arm_structure()
    interpolate_density()
    interpolate_local_arm_density()

if __name__ == '__main__':
    main()