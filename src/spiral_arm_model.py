import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import gc
import os
import logging
logging.basicConfig(level=logging.INFO)     # other levels for future reference: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
import src.observational_data.firas_data as firas_data
import src.utilities.utilities as ut
import src.utilities.constants as const
import src.utilities.settings as settings
#from galaxy_tests import test_plot_density_distribution



def spiral_arm_medians(arm_angle, pitch_angle, rho_min=const.rho_min_spiral_arm, rho_max=const.rho_max_spiral_arm):
    """ Function to calculate the medians of the spiral arms. The medians are calculated in polar coordinates.
    Args:
        arm_angle (int): starting angle of the spiral arm, radians
        pitch_angle (int): pitch angle of the spiral arm, radians
        rho_min (float): minimum distance from the Galactic center. Units of kpc
        rho_max (float): maximum distance from the Galactic center. Units of kpc

    Returns:
        theta and rho for the spiral arm medians
    """
    
    theta = [arm_angle]
    rho = [rho_min]
    dtheta = .01
    k = np.tan(pitch_angle)
    while rho[-1] < rho_max: # Keep adding points until the last point is at the maximum allowed distance from the Galactic center
        theta.append((theta[-1] + dtheta))
        rho.append(rho_min * np.exp(k * (theta[-1] - theta[0]))) # Equation 6 in Higdon and Lingenfelter
    return np.array(theta), np.array(rho)


def arm_transverse_density(transverse_distances, sigma_arm=const.sigma_arm):
    """ Function to calculate the fall off of spiral arm populations transverse an arm median
    Args:
        transverse_distances (np.array): transverse distances from the medians of the modelled spiral arm
        sigma_arm (float): dispersion of the spiral arm. Units of kpc

    Returns:
        the fall off of spiral arm populations transverse an arm median
    """
    return np.exp(-0.5 * transverse_distances ** 2 / sigma_arm ** 2)  / (np.sqrt(2 * np.pi) * sigma_arm) # in the paper, they do not include this normalization factor for some reason


def generate_transverse_spacing_densities(sigma_arm=const.sigma_arm, d_min=0.01, d_max=5, scaling=0.1):
    """ Function to generate non-uniform spacing for the transverse distances from the spiral arm median
    Args:
        sigma_arm (float): dispersion of the spiral arm. Units of kpc.
        d_min (float): minimum distance from the spiral arm. Units of kpc. 
        d_max (float): maximum distance from the spiral arm. Units of kpc.
        scaling: scaling of the exponential distribution. Determines how many points are generated. 

    Returns:
        transverse_distances (1D array) with the relative distances for the transverse parts of the spiral arms. First element is d_min, last element is d_max.
        transverse_densities (1D array) with the fall off in density for the transverse distances.
    """
    d_r = np.array([d_min]) # d_rho is an array with the increments d_r in the radial direction from the spiral arm median
    while np.sum(d_r) < d_max: # keep adding points until the sum of drho is larger than d_max
        d_r = np.append(d_r, np.random.exponential(scale=scaling) + d_min) # + d_min ensures that the minimum distance from the spiral arm is d_min
    # now the sum of drho is larger than 5, so we need to adjust for that
    diff = np.sum(d_r) - (d_max) # How much we have overshot the maximum distance
    d_r.sort() # sort the array in ascending order
    d_r[-1] = d_r[-1] - diff # adjust the last (largest) element in the array so that the sum of drho is exactly 5
    d_r.sort() # sort the array in ascending order
    transverse_distances = np.cumsum(d_r) # get the actual transverse distances from the spiral arm median for every point in the array (and not just the relative distance between every point, which is what d_r is)
    transverse_densities = arm_transverse_density(transverse_distances, sigma_arm) # calculate the fall off of spiral arm populations transverse an arm median
    return transverse_distances, transverse_densities # return the transverse points and fall off in density


def generate_end_points(rho, theta, pitch_angle, transverse_distances, point='start'):
    """
    Args:
        rho (float): radial distance to one of the end points of the spiral arm. Single number
        theta (float): angular distance to one of the end points of the spiral arm. Single number
        pitch_angle (float): pitch angle of the spiral arm. Single number
        transverse_distances (1D array): transverse distances from the spiral arm median
        point (str): 'start' or 'end'. Indicates if the end point is the start or the end of the spiral arm.

    Returns:
        x_arc (1D array): x-coordinates for the circular projection around the end points of the spiral arms
        y_arc (1D array): y-coordinates for the circular projection around the end points of the spiral arms
    """
    angles_arc = np.linspace(0, np.pi, num=const.number_of_end_points) + theta - pitch_angle # the angles for the circular projection around the end points of the spiral arms
    if point == 'start':
        angles_arc += np.pi # adjust the angles so that the circular projection is around the start point of the spiral arm
    x_arc = rho * np.cos(theta) + transverse_distances * np.cos(angles_arc)[:, np.newaxis]
    y_arc = rho * np.sin(theta) + transverse_distances * np.sin(angles_arc)[:, np.newaxis]
    return x_arc.ravel(), y_arc.ravel()


def generate_spiral_arm_coordinates(arm_medians, transverse_distances, thetas, pitch_angle):
    """ Function to generate the coordinates for the spiral arm. The coordinates are in xy coordinates
    Args:
        arm_medians: array of distances rho for the arm median
        transverse_distances: array of transverse distances from the arm median
        thetas: array of thetas for the arm median
        pitch_angle: pitch angle of the spiral arm

    Returns:
        x_spiral_arm: 1D array with the x-coordinates for the spiral arm
        y_spiral_arm: 1D array with the y-coordinates for the spiral arm
    """
    # calculate the cosinus and sinus values for the angles
    angle_cos = np.cos(thetas - pitch_angle)
    angle_sin = np.sin(thetas - pitch_angle)
    # with these angles, calculate the transverse points in xy coordinates. 
    x_transverse = angle_cos[:, np.newaxis] * transverse_distances
    y_transverse = angle_sin[:, np.newaxis] * transverse_distances
    # so that we have the transverse points on both sides of the spiral arm, we need to flip the transverse points and concatenate them
    x_transverse_mirrored = -np.flip(x_transverse, axis=1)
    y_transverse_mirrored = -np.flip(y_transverse, axis=1)
    zeros_column = np.zeros((len(arm_medians), 1)) # the zeros column is used to represent the arm median
    x_spiral_arm = np.concatenate((x_transverse_mirrored, zeros_column, x_transverse), axis=1)
    y_spiral_arm = np.concatenate((y_transverse_mirrored, zeros_column, y_transverse), axis=1)
    # now we need to move the transverse points out to the correct rho
    x_arms_medians = arm_medians * np.cos(thetas)
    y_arms_medians = arm_medians * np.sin(thetas)
    # Add the arm medians to the transverse points. This is done by broadcasting the arm medians to the shape of the transverse points. 
    x_spiral_arm += x_arms_medians[:, np.newaxis]
    y_spiral_arm += y_arms_medians[:, np.newaxis]

    start_x, start_y = generate_end_points(arm_medians[0], thetas[0], pitch_angle, transverse_distances, 'start')
    end_x, end_y = generate_end_points(arm_medians[-1], thetas[-1], pitch_angle, transverse_distances, 'end')
    x_spiral_arm = np.concatenate((start_x, x_spiral_arm.ravel(), end_x))
    y_spiral_arm = np.concatenate((start_y, y_spiral_arm.ravel(), end_y))
    return x_spiral_arm, y_spiral_arm


def generate_spiral_arm_densities(rho, transverse_densities_initial, h=const.h_spiral_arm):
    """ Function to calculate the densities for the spiral arm. The densities are calculated for the spiral arm median, the transverse points and the circular projection at the end points
    Args:
        rho: 1D array of radial distances to the spiral arm median
        transverse_densities_initial: 1D array of transverse distances from the spiral arm median. Initial simply indicates that these are the distances on one side of the spiral arm median
        h: scale length of the disk.
    
    Returns:
        density_spiral_arm: 1D array of densities for the spiral arm. The densities appears in the same order as the spiral arm points in rho_coords and theta_coords as returned by generate_spiral_arm_coordinates
    """
    transverse_densities = np.append(arm_transverse_density(transverse_distances = 0), transverse_densities_initial) # the 1 is to take into account the arm median itself. SHOULD INSTEAD OF USING 1 TRY TO USE arm_transverse_density(transverse_distances = 0, sigma_arm) HERE
    transverse_densities = np.append(np.flip(transverse_densities_initial), transverse_densities) # so that we have the transverse densities on both sides of the spiral arm
    # calculate the densities for the arm median
    arm_median_densities = ut.axisymmetric_disk_population(rho, h) #1D array
    # calculate the transverse densities for the arm. Does not contain the contrbution from the circular projection at the end points, but has the arm median density
    # note the transverse densities are described by a Gaussian distribution relative to the arm median
    arm_transverse_densities = transverse_densities * arm_median_densities[:, np.newaxis] #2D array
    # calculate the densities for the end points projected in a circle around the end points
    # Note the use of transverse_densities_initial: this is because the density for each line of points in the circular projection are equal to each other. 
    # Also, it is important to note that the arm median density for the end points are not included in the circular projection, as the arm median would be repeated number_of_end_points times
    density_start_arm = transverse_densities_initial * arm_median_densities[0] # this is a 1D array, but the same values goes for every index in start_arm along axis 0
    density_end_arm = transverse_densities_initial * arm_median_densities[-1] # this is a 1D array, but the same values goes for every index in end_arm along axis 0
    density_spiral_arm = np.concatenate([np.tile(density_start_arm, const.number_of_end_points), arm_transverse_densities.flatten(), np.tile(density_end_arm, const.number_of_end_points)])
    return density_spiral_arm


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
    transverse_distances, transverse_densities_initial = generate_transverse_spacing_densities(sigma_arm) 
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy')
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy')
    num_grid_subdivisions = settings.num_grid_subdivisions
    if num_grid_subdivisions < 1:
        raise ValueError("num_grid_subdivisions must be larger than 0")
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i], const.rho_min_spiral_arm[i], const.rho_max_spiral_arm[i])
        # generate the spiral arm points
        x, y = generate_spiral_arm_coordinates(rho, transverse_distances, theta, pitch_angles[i])
        # generate the spiral arm densities
        density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial, h)
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
            np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{i}_{sub_grid}.npy', interpolated_arm)
    return


def calc_effective_area_per_spiral_arm(h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles):
    """ Function to calculate the effective area for each spiral arm. The density of each spiral arm is integrated over the entire galactic plane.
    The returned effective areas are in units of kpc^2, and appears in the same order as the spiral arms in arm_angles.

    Args:
        h (float, optional): Scale length of the disk. Defaults to h_spiral_arm.
        sigma_arm (float, optional): Dispersion of the spiral arms. Defaults to sigma_arm.
        arm_angles (list, optional): Starting angles for the spiral arms. Defaults to arm_angles.
        pitch_angles (list, optional): Pitch angles for the spiral arms. Defaults to pitch_angles.

    Returns:
        1D np.array: Effective area for each spiral arm. The effective areas appears in the same order as the spiral arms in arm_angles.
        Also saves the effective areas to a file
    """
    logging.info("Calculating the effective area for each spiral arm")
    transverse_distances, transverse_densities_initial = generate_transverse_spacing_densities(sigma_arm) 
    d_x = 70 / 3000 # distance between each interpolated point in the x direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    d_y = 70 / 3000 # distance between each interpolated point in the y direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    grid_x, grid_y = np.mgrid[-35:35:3000j, -35:35:3000j]
    effective_area = []
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i], const.rho_min_spiral_arm[i], const.rho_max_spiral_arm[i])
        # generate the spiral arm points
        x, y = generate_spiral_arm_coordinates(rho, transverse_distances, theta, pitch_angles[i])
        # generate the spiral arm densities
        density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial, h)
        # calculate interpolated density for the spiral arm
        interpolated_density = griddata((x, y), density_spiral_arm, (grid_x, grid_y), fill_value=0)
        interpolated_density[interpolated_density < 0] = 0 # set all negative values to 0
        # add the interpolated density to the total galactic density
        effective_area = np.append(effective_area, np.sum(interpolated_density) * d_x * d_y)
    filepath = f'{const.FOLDER_GALAXY_DATA}/effective_area_per_spiral_arm.npy'
    np.save(filepath, effective_area)
    return effective_area


def generate_latitudinal_points(b_max=5.0, db_above_1_deg = 0.2, b_min=0.01, b_lim_exp=1, scaling=0.015):
    """ Function to generate latitudinal points. The latitudinal points are generated in such a way that the density of points is higher close to the Galactic plane.
    Args:
        b_min: minimum angular distance from the plane
        b_lim_exp: limit on angular distance from plane for the exponential distribution
        b_max: maximum angular distance from the plane
        scaling: scaling of the exponential distribution. A value of 0.015 generates between 110 - 130 points. Larger scaling = fewer points
    Returns:
        1D array with the angular distances from the Galactic plane, and an array with all increments (db) between the points
        Arrays made ready to be used in a central Rieman sum 
    """
    
    db = np.array([b_min])
    # in case one wants to have a b_max lower than 1 degree (i.e. b_lim_exp), we need to adjust b_lim_exp
    if b_max < b_lim_exp:
        b_lim_exp = b_max

    while np.sum(db) < b_lim_exp: # to make sure that the first element shall be b_min
        db = np.append(db, np.random.exponential(scale=scaling) + b_min)
    # now the sum of db is larger than b_max, so we need to adjust for that
    diff = np.sum(db) - (b_lim_exp)
    db.sort()
    db[-1] = db[-1] - diff
    db.sort()
    latitudes = np.cumsum(db)
    # move each dr, such that they are centralized on each point. Essential for the Riemann sum
    dr_skewed = np.append(db[1:], db_above_1_deg / 2) # / 2 because this would be the width between 1.0 and the next point. 
    db = (db + dr_skewed) / 2  
    # transverse_distances done for b_min <= b <= b_max. Now we need to do b_max < b <= 5
    if latitudes[0] == b_min: # to make sure that the first element is be b_min
        ang_dist_above_b_lim_exp = b_max - b_lim_exp
        if db_above_1_deg == 0 or b_max == 1:
            latitudes = np.concatenate((-latitudes[::-1], [0], latitudes))
            db = np.concatenate((db[::-1], [b_min], db))
            np.save(f'{const.FOLDER_GALAXY_DATA}/latitudes.npy', np.radians(latitudes)) # saving latitudes
            np.save(f'{const.FOLDER_GALAXY_DATA}/db.npy', np.radians(db)) # saving db
            return
        elif ang_dist_above_b_lim_exp / db_above_1_deg == int(ang_dist_above_b_lim_exp / db_above_1_deg): # to make sure that the number of latitudinal angles between b_lim_exp and b_max is an integer 
            # add the first point above b_lim_exp, since it will have a different width than the rest of the points above b_lim_exp
            latitudes = np.append(latitudes, latitudes[-1] + db_above_1_deg / 2)
            db = np.append(db, db_above_1_deg * 0.75) # same reason as for dr_skewed: the total with between 1.0 and the first point to its right is half that of db_above_1_deg
            print("int(ang_dist_above_b_lim_exp / db_above_1_deg)", int(ang_dist_above_b_lim_exp / db_above_1_deg))
            for _ in range(1, int(ang_dist_above_b_lim_exp / db_above_1_deg)):
                latitudes = np.append(latitudes, latitudes[-1] + db_above_1_deg)
                db = np.append(db, db_above_1_deg)
            latitudes = np.concatenate((-latitudes[::-1], [0], latitudes))
            db = np.concatenate((db[::-1], [b_min], db))
            # latitudes and db are now ready to be used in a central Riemann sum
            # note that, due to this being a central Riemann sum, the last element in latitudes is b_max - db_above_1_deg / 2. The edge of the last bin is at b_max
            np.save(f'{const.FOLDER_GALAXY_DATA}/latitudes.npy', np.radians(latitudes)) # saving latitudes
            np.save(f'{const.FOLDER_GALAXY_DATA}/db.npy', np.radians(db)) # saving db
            return 
        else:
            raise ValueError("The number of latitudinal angles is not an integer. Please change the value of db_above_1_deg")
    else:
        print("The first element of transverse_distances is not b_min. Trying again...")
        return generate_latitudinal_points(b_max, db_above_1_deg, b_min, b_lim_exp, scaling)
    

def calculate_galactic_coordinates(b_max=5, db_above_1_deg = 0.2, b_min=0.01, b_lim_exp=1, scaling=0.015):
    logging.info("Calculating the galactic coordinates")
    # Calculate coordinates
    dr = 0.2   # increments in dr (kpc):
    dl = 0.2    # increments in dl (degrees):
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    dl = np.radians(dl)
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(dr, const.r_s + const.rho_max_spiral_arm[0] + 5 + dr, dr) # +5 kpc to take into account the width of the spiral arms
    # save coordinates to disk
    np.save(f'{const.FOLDER_GALAXY_DATA}/radial_distances.npy', radial_distances) # saving radial_distances
    np.save(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy', longitudes) # saving longitudes
    np.save(f'{const.FOLDER_GALAXY_DATA}/dr.npy', dr) # saving dr
    np.save(f'{const.FOLDER_GALAXY_DATA}/dl.npy', dl) # saving dl
    generate_latitudinal_points(b_max, db_above_1_deg, b_min, b_lim_exp, scaling)
    latitudes = np.load(f'{const.FOLDER_GALAXY_DATA}/latitudes.npy')
    logging.info("Length of latitudes: " + str(len(latitudes)) + ". Length of radial_distances: " + str(len(radial_distances)) + ". Length of longitudes: " + str(len(longitudes)) + ".")
    logging.info("Max latitude: " + str(np.degrees(latitudes[-1])) + ". Min latitude: " + str(np.degrees(latitudes[0])) + ".")
    logging.info("Creating coordinate grids")
    radial_grid, long_grid, lat_grid = np.meshgrid(radial_distances, longitudes, latitudes, indexing='ij')
    logging.info("Shape of radial_grid: " + str(radial_grid.shape) + ". Shape of long_grid: " + str(long_grid.shape) + ". Shape of lat_grid: " + str(lat_grid.shape) + ".")
    np.save(f'{const.FOLDER_GALAXY_DATA}/radial_grid.npy', radial_grid.ravel()) # saving radial_grid
    np.save(f'{const.FOLDER_GALAXY_DATA}/long_grid.npy', long_grid.ravel()) # saving long_grid
    np.save(f'{const.FOLDER_GALAXY_DATA}/lat_grid.npy', lat_grid.ravel()) # saving lat_grid
    del radial_grid, long_grid, lat_grid, latitudes, longitudes, radial_distances, dr, dl
    gc.collect()
    logging.info("Grids created. Now calculating the latidudinal cosinus")
    latitudinal_cosinus = np.cos(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/lat_grid.npy'))
    np.save(f'{const.FOLDER_GALAXY_DATA}/latitudinal_cosinus.npy', latitudinal_cosinus) # saving latitudinal_cosinus
    del latitudinal_cosinus
    gc.collect()
    logging.info("Latitudinal cosinus calculated. Now calculating z_grid and height_distribution_values")
    z_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/radial_grid.npy') * np.sin(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/lat_grid.npy'))
    np.save(f'{const.FOLDER_GALAXY_DATA}/z_grid.npy', z_grid) # saving z_grid
    del z_grid
    gc.collect()
    height_distribution_values = ut.height_distribution(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/z_grid.npy'), sigma=const.sigma_height_distr)
    np.save(f'{const.FOLDER_GALAXY_DATA}/height_distribution_values.npy', height_distribution_values) # saving height_distribution_values
    del height_distribution_values
    gc.collect()
    logging.info("z_grid and height_distribution_values calculated. Now calculating rho, theta, x and y coordinates")
    rho_coords_galaxy = ut.rho(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/radial_grid.npy'), np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/long_grid.npy'), np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/lat_grid.npy'))
    np.save(f'{const.FOLDER_GALAXY_DATA}/rho_coords_galaxy.npy', rho_coords_galaxy) # saving rho_coords_galaxy
    del rho_coords_galaxy
    gc.collect()
    logging.info("rho coordinates calculated")
    theta_coords_galaxy = ut.theta(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/radial_grid.npy'), np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/long_grid.npy'), np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/lat_grid.npy'))
    np.save(f'{const.FOLDER_GALAXY_DATA}/theta_coords_galaxy.npy', theta_coords_galaxy) # saving theta_coords_galaxy
    del theta_coords_galaxy
    gc.collect()
    logging.info("theta coordinates calculated. Now removing the grid files from disk")
    os.remove(f'{const.FOLDER_GALAXY_DATA}/radial_grid.npy')
    os.remove(f'{const.FOLDER_GALAXY_DATA}/long_grid.npy')
    os.remove(f'{const.FOLDER_GALAXY_DATA}/lat_grid.npy')
    logging.info("Calculating x-values")
    x_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/rho_coords_galaxy.npy') * np.cos(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/theta_coords_galaxy.npy'))
    np.save(f'{const.FOLDER_GALAXY_DATA}/x_grid.npy', x_grid) # saving x_grid
    del x_grid
    gc.collect()
    logging.info("Calculating y-values")
    y_grid = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/rho_coords_galaxy.npy') * np.sin(np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/theta_coords_galaxy.npy'))
    np.save(f'{const.FOLDER_GALAXY_DATA}/y_grid.npy', y_grid) # saving y_grid
    del y_grid
    gc.collect()
    # delte from disk the rho_coords_galaxy, theta_coords_galaxy:
    logging.info("x and y coordinates calculated. Now removing the rho and theta coordinates from disk")
    os.remove(f'{const.FOLDER_GALAXY_DATA}/theta_coords_galaxy.npy')
    # need the rho_grids for the axisymmetric model
    return
    

def calc_modelled_emissivity(b_max=1, db_above_1_deg = 0.1, fractional_contribution=const.fractional_contribution, readfile_effective_area=True, h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles):
    logging.info("Calculating modelled emissivity of the Milky Way")
    if readfile_effective_area == True:
        effective_area = np.load(f'{const.FOLDER_GALAXY_DATA}/effective_area_per_spiral_arm.npy')
    elif readfile_effective_area == False:
        effective_area = calc_effective_area_per_spiral_arm(h, sigma_arm, arm_angles, pitch_angles)
    else:
        raise ValueError("readfile_effective_area must be either True or False")
    print(effective_area)
    #calculate_galactic_coordinates(b_max, db_above_1_deg)
    logging.info("Coordinates calculated. Now interpolating each spiral arm")
    # coordinates made. Now we need to interpolate each spiral arm and sum up the densities
    #interpolate_density(h, sigma_arm, arm_angles, pitch_angles)
    logging.info("Interpolation done. Now calculating the emissivity")
    common_multiplication_factor = const.total_galactic_n_luminosity * np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/height_distribution_values.npy')
    for i in range(len(arm_angles)): # loop trough the 4 spiral arms
        logging.info(f"Calculating emissivity for spiral arm number: {i+1}")
        scaled_arm_emissivity = np.load(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{i}_0.npy') 
        for j in range(1, settings.num_grid_subdivisions): # loop through the different grid subdivisions
            scaled_arm_emissivity = np.concatenate((scaled_arm_emissivity, np.load(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_{i}_{j}.npy')))
        scaled_arm_emissivity *= common_multiplication_factor * fractional_contribution[i] / (effective_area[i] * const.kpc**2) # multiply with the factors that makes the emissivity in units of erg/s/cm^2/sr
        # save the emissivity for the arm to disk
        np.save(f'{const.FOLDER_GALAXY_DATA}/interpolated_arm_emissivity_{i}.npy', scaled_arm_emissivity)
    return  


def calc_modelled_intensity(b_max=5, db_above_1_deg = 0.2, fractional_contribution=const.fractional_contribution, readfile_effective_area=True, h=const.h_spiral_arm, sigma_arm=const.sigma_arm, arm_angles=const.arm_angles, pitch_angles=const.pitch_angles):
    logging.info("Calculating the modelled NII intensity of the Milky Way")
    calc_modelled_emissivity(b_max, db_above_1_deg, fractional_contribution, readfile_effective_area, h, sigma_arm, arm_angles, pitch_angles)
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
    intensities_per_arm = np.zeros((len(arm_angles), num_longs)) # to store the intensity as a function of longitude for each spiral arm. Used for the intesnity-plots to compare with Higdon & Lingenfelter
    for i in range(len(arm_angles)):
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


def plot_modelled_intensity_per_arm(filename_output = f'{const.FOLDER_MODELS_GALAXY}/modelled_intensity.pdf', filename_intensity_data = f'{const.FOLDER_GALAXY_DATA}/intensities_per_arm_b_max_5.npy', fractional_contribution=const.fractional_contribution, h=const.h_spiral_arm, sigma_arm=const.sigma_arm):
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
    intensities_per_arm = np.lib.format.open_memmap(filename_intensity_data)
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[0], label=f"NC. f={fractional_contribution[0]}")
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[1], label=f"P. $\ $ f={fractional_contribution[1]}")
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[2], label=f"SA. f={fractional_contribution[2]}")
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[3], label=f"SC. f={fractional_contribution[3]}")
    if settings.add_local_arm_to_intensity_plot == True:
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[4], label=f"Local arm. f={fractional_contribution[4]}")
        plt.plot(np.linspace(0, 360, len(longitudes)), np.sum(intensities_per_arm, axis=0), label="Total")
    else:
        plt.plot(np.linspace(0, 360, len(longitudes)), np.sum(intensities_per_arm[:4], axis=0), label="Total")
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Line intensity in erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$")
    plt.title("Modelled intensity of the Galactic disk")
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.9, fr'NII Luminosity = {const.total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig(filename_output)
    plt.close()


def test_max_b():
    """ Function to test the effect of changing the maximum angular distance from the Galactic plane on the modelled intensity of the Milky Way"""
    b_max = np.array([0.5, 1.0, 3.5, 5.0])
    db_above_1_deg = np.array([0, 0, 0.1, 0.2])
    for i in range(len(b_max)):
        calc_modelled_intensity(b_max=b_max[i], db_above_1_deg=db_above_1_deg[i])
        b_filename = str(b_max[i]).replace(".", "_")
        filename_output = f'{const.FOLDER_MODELS_GALAXY}/modelled_intensity_b_max_{b_filename}.pdf'
        filename_intensity_data = f'{const.FOLDER_GALAXY_DATA}/intensities_per_arm_b_max_{b_filename}.npy'
        plot_modelled_intensity_per_arm(filename_output, filename_intensity_data)


def main() -> None:
    logging.info("Starting main function")
    calc_modelled_intensity(readfile_effective_area=True)
    plot_modelled_intensity_per_arm()
    #test_max_b()


if __name__ == "__main__":
    main()

# SO the pitch angles must also be changed. 
# For SA: the larger starting angle, the closer to GC the luminocity is concentrated. Will thus adopt a starting angle of 245 degrees for next test run
# For SC: the larger starting angle, the closer to GC the luminocity is concentrated. Will thus adopt a starting angle of 335 degrees for next test run
# For P: the smaller the starting angle, the closer to GC the luminocity is concentrated. Also, the height of the peak is reduced by about 20% between the smallest and largest starting angle. Will keep the starting angle at 160 degrees for next test run
# For NC: the higher the value, the more right-skewed the luminocity distribution becomes. Will set the starting angle at 65 degrees for next test run. Also, a larger angle makes the "dump" in the middle les dumpy, and vice versa.
