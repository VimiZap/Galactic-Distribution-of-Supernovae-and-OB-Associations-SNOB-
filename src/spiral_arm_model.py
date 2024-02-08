import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import gc
import time
import os
import logging
import observational_data.firas_data as firas_data
#from galaxy_tests import test_plot_density_distribution



def test_path():
    work_directory = '/work/paradoxx/viktormi/output'
    filename = 'example.npy'
    filepath = os.path.join(work_directory, filename)
    data = np.arange(0, 10, 1)
    np.save(filepath, data)

# constants
h_default = 2.4                 # kpc, scale length of the disk. The value Higdon and Lingenfelter used
r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
# This rho_max and rho_min seems to be taken from Valee
rho_min = 2.9           # kpc, minimum distance from galactic center to the beginning of the spiral arms.
rho_max = 35            # kpc, maximum distance from galactic center to the end of the spiral arms.
sigma = 0.15            # kpc, scale height of the disk
sigma_arm_default = 0.5         # kpc, dispersion of the spiral arms
total_galactic_n_luminosity = 1.4e40 # / (np.pi/2)     #total galactic N 2 luminosity in erg/s
gum_nii_luminosity = 1e36 # erg/s, luminosity of the Gum Nebula in N II 205 micron line. Number from Higdon and Lingenfelter
cygnus_nii_luminosity = 2.4e37 # erg/s, luminosity of the Cygnus Loop in N II 205 micron line. Number from Higdon and Lingenfelter
measured_nii = 1.175e-4 # erg/s/cm^2/sr, measured N II 205 micron line intensity. Estimated from the graph in the paper
kpc = 3.08567758e21    # 1 kpc in cm
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
# note that the authors where this formula came from used different numbers, and got a value of 47 kpc^2.
# With these numbers, we get a value of 22.489 kpc^2
a_d = 2*np.pi*h_default**2 * ((1+rho_min/h_default)*np.exp(-rho_min/h_default) - (1+rho_max/h_default)*np.exp(-rho_max/h_default)) 
# starting angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
#arm_angles = np.radians([70, 160, 250, 340]) #original
#arm_angles = np.radians([80, 170, 260, 350]) #30
#arm_angles = np.radians([72, 162, 252, 342]) #31
#arm_angles = np.radians([60, 150, 240, 330]) #32, 33
arm_angles = np.radians([65, 160, 240, 330])  # best fit for the new r_s

# pitch angles for the spiral arms, respectively Norma-Cygnus(NC), Perseus(P), Sagittarius-Carina(SA), Scutum-Crux(SC)
#pitch_angles = np.radians(np.array([13.5, 13.5, 13.5, 15.5])) #original
pitch_angles = np.radians([14, 14, 14, 16]) # best fir to new r_s

# the fractional contribution described in the text.
fractions = [0.18, 0.36, 0.18, 0.28]
number_of_end_points = 45 # number of points to use for the circular projection at the end points of the spiral arms
spiral_arm_names = ['Norma-Cygnus', 'Perseus', 'Sagittarius-Carina', 'Scutum-Crux']
fractional_contribution_default = [0.18, 0.36, 0.18, 0.28] # [0.17, 0.34, 0.15, 0.34]


def count_negative_values(arr):
    negative_values = arr[arr < 0]
    return len(negative_values), np.average(negative_values)


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


def spiral_arm_medians(arm_angle, pitch_angle):
    """
    Args:
        arm_angle: starting angle of the spiral arm, radians
        pitch_angle: pitch angle of the spiral arm, radians
    Returns:
        values for thetas and the corresponding rhos for the spiral arm
    """
    
    theta = [arm_angle]
    rho = [rho_min]
    dtheta = .01
    k = np.tan(pitch_angle)
    while rho[-1] < rho_max:
        theta.append((theta[-1] + dtheta) )#% (2*np.pi)) # the % is to make sure that theta stays between 0 and 2pi
        rho.append(rho_min*np.exp(k*(theta[-1] - theta[0])))
    
    print("Number of points for the given spiral arm: ", len(theta))
    return np.array(theta), np.array(rho)


def height_distribution(z): # z is the height above the Galactic plane
    """
    Args:
        z: height above the Galactic plane
    Returns:
        height distribution of the disk at a height z above the Galactic plane
    """
    return np.exp(-0.5 * z**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def arm_transverse_density(delta, sigma_arm):
    """
    Args:
        delta: transverse distance from the medians of the modelled spiral arm
        sigma_arm: dispersion of the spiral arm
    Returns:
        the fall off of spiral arm populations transverse an arm median
    """
    return np.exp(-0.5 * delta**2 / sigma_arm**2)  / (np.sqrt(2*np.pi) * sigma_arm) # in the paper, they do not include this normalization factor for some reason


def arm_median_density(rho, h=h_default): 
    """
    Args:
        rho: distance from the Galactic center
        h: scale length of the disk
    Returns:
        density of the disk at a distance rho from the Galactic center
    """
    return np.exp(-rho/h)


def generate_non_uniform_spacing(sigma_arm=sigma_arm_default, d_min=0.01, d_max=5, scaling=0.1):
    """
    Args:
        d_min: minimum distance from the spiral arm
        d_max: maximum distance from the spiral arm
        scaling: scaling of the exponential distribution. A value of 0.03 generates between 150-200 points, for the most part 160-180 points
    Returns:
        1D array with the relative distances for the transverse parts of the spiral arms in radial units rho. First element is d_min, last element is d_max
    """
    d_rho = np.array([d_min])
    while np.sum(d_rho) < d_max: # to make sure that the dirst dx shall be d_min
        d_rho = np.append(d_rho, np.random.exponential(scale=scaling) + d_min)
    # now the sum of drho is larger than 5, so we need to adjust for that
    diff = np.sum(d_rho) - (d_max)
    d_rho.sort()
    d_rho[-1] = d_rho[-1] - diff
    d_rho.sort()
    transverse_distances = np.cumsum(d_rho)
    transverse_densities = arm_transverse_density(transverse_distances, sigma_arm)
    return transverse_distances, transverse_densities


def generate_transverse_points(arm_medians, transverse_distances, thetas, pitch_angle):
    """
    Args:
        arm_medians: array of distances \rho to the arm median
        transverse_distances: array of transverse distances from the arm median
        thetas: array of thetas for the arm median
        pitch_angle: pitch angle of the spiral arm
    Returns:
        a 3d array, where the first index is a given point on the spiral arm, the second index is the transverse point, and the third index is the x and y coordinates
    """
    # calculate the cosinus and sinus values for the angles
    angle_cos = np.cos(thetas - pitch_angle)
    angle_sin = np.sin(thetas - pitch_angle)
    # with these angles, calculate the transverse points in xy coordinates. 3D array: the first index is the point on the spiral arm, the second index is the transverse point, and the third index is the x and y coordinates
    x_rotated_transverse = angle_cos[:, np.newaxis] * transverse_distances
    y_rotated_transverse = angle_sin[:, np.newaxis] * transverse_distances
    rotated_transverse = np.concatenate((x_rotated_transverse[:, :, np.newaxis], y_rotated_transverse[:, :, np.newaxis]), axis=2) #3D array is created here
    # so that we have the transverse points on both sides of the spiral arm, we need to flip the transverse points and concatenate them
    rotated_transverse_negatives = -np.flip(rotated_transverse, axis=1)
    rotated_transverse_total = np.append(rotated_transverse_negatives, rotated_transverse, axis=1)
    # now we need to move the transverse points out to the correct rho
    x_arms_medians = arm_medians * np.cos(thetas)
    y_arms_medians = arm_medians * np.sin(thetas)
    x_y_arms_medians = np.array([x_arms_medians, y_arms_medians]).T
    final_transverse_points = rotated_transverse_total + x_y_arms_medians[:, np.newaxis, :]
    return final_transverse_points


def generate_end_points(rho, theta, pitch_angle, transverse_distances, point='start'):
    """
    Args:
        rho: radial distance to one of the end points of the spiral arm
        theta: angular distance to one of the end points of the spiral arm
    Returns:
        a 3D array with the xy coordinates for the half-sphere around the end-point of the spiral arm
        axis = 0 is basically an index for the angle of the circular projection
        axis = 1 is an index for the points on the circular projection
        axis = 2 is an index for the x and y coordinates
    """
    angles_arc = np.linspace(0, np.pi, num=number_of_end_points) + theta - pitch_angle 
    if point == 'start':
        angles_arc += np.pi
    x_arc = rho * np.cos(theta) + transverse_distances * np.cos(angles_arc)[:, np.newaxis]
    y_arc = rho * np.sin(theta) + transverse_distances * np.sin(angles_arc)[:, np.newaxis]
    return np.concatenate((x_arc[:, :, np.newaxis], y_arc[:, :, np.newaxis]), axis=2)


def plot_spiral_arms(sigma_arm=sigma_arm_default):
    """
    Plots the spiral arms, both the medians and also the transverse points. 
    """
    plt.scatter(0,0, c='magenta')
    colours = ['navy', 'darkgreen', 'darkorange', 'purple'] # one colour per arm
    transverse_distances, transverse_densities = generate_non_uniform_spacing(sigma_arm) #d_min: minimum distance from the spiral arm
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i])
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        # generate the transverse points
        transverse_points = generate_transverse_points(rho, transverse_distances, theta, pitch_angles[i])
        # Flatten the 3D array into 2 1D arrays
        x_coords = transverse_points[:, :, 0].flatten()
        y_coords = transverse_points[:, :, 1].flatten()
        # generate the end points
        start_arm = generate_end_points(rho[0], theta[0], pitch_angles[i], transverse_distances, 'start')
        end_arm = generate_end_points(rho[-1], theta[-1], pitch_angles[i], transverse_distances, 'end')
        colour = np.linspace(0, 1, len(x_coords)) # this generates a colour gradient along the spiral arm
        plt.scatter(x_coords, y_coords, s=1, c=colours[i])  # You can adjust the marker size (s) as needed
        plt.scatter(start_arm[:, :, 0].flatten(), start_arm[:, :, 1].flatten(), s=1, c=colours[i])
        plt.scatter(end_arm[:, :, 0].flatten(), end_arm[:, :, 1].flatten(), s=1, c=colours[i])
        plt.plot(x, y)
    plt.gca().set_aspect('equal')
    plt.savefig("output/spiral_arms_2_w_transverse_contribution.png")  # save plot in the output folder
    plt.show()


def generate_spiral_arm_points_spherical_coords(rho, theta, pitch_angle, transverse_distances):
    """
    Args:
        rho: 1D array of radial distances to the spiral arm median
        theta: 1D array of angular distances to the spiral arm median
        pitch_angle: pitch angle of the spiral arm
        transverse_distances: 1D array of transverse distances from the spiral arm median
    Returns:
        rho_coords, theta_coords
        for a given spiral arm, the spherical coordinates for the spiral arm are returned. Includes arm median, the transverse points and the circular projection at the end points
        rho_coords, theta_coords are 1D arrays
    """
    # generate the transverse points. This is a 3D array: the first index is the point on the spiral arm, the second index is the transverse point, and the third index is the x and y coordinates
    transverse_points = generate_transverse_points(rho, transverse_distances, theta, pitch_angle)
    # convert the transverse points to spherical coordinates, and insert the arm medians rho and theta in the middle of the arrays
    pos_arm_median = len(transverse_points[0]) // 2
    radial_transverse_points = np.sqrt(transverse_points[:, :, 0]**2 + transverse_points[:, :, 1]**2)
    radial_transverse_points = np.insert(radial_transverse_points, pos_arm_median, rho, axis=1)
    angular_transverse_points = np.arctan2(transverse_points[:, :, 1], transverse_points[:, :, 0])
    angular_transverse_points = np.insert(angular_transverse_points, pos_arm_median, theta, axis=1)
    # generate the end points for the spiral arm. 3D array, axis 0 is basically the angle for the circular projection, axis 1 are the points, axis 2 are the x and y coordinates
    start_arm = generate_end_points(rho[0], theta[0], pitch_angle, transverse_distances, 'start')
    end_arm = generate_end_points(rho[-1], theta[-1], pitch_angle, transverse_distances, 'end')
    # get the endpoints in spherical coordinates
    radial_start_arm = np.sqrt(start_arm[:, :, 0]**2 + start_arm[:, :, 1]**2)
    angular_start_arm = np.arctan2(start_arm[:, :, 1], start_arm[:, :, 0])
    radial_end_arm = np.sqrt(end_arm[:, :, 0]**2 + end_arm[:, :, 1]**2)
    angular_end_arm = np.arctan2(end_arm[:, :, 1], end_arm[:, :, 0])
    # now we need to gather the arm and endpoints together
    rho_coords = np.append(radial_start_arm.flatten(), radial_transverse_points.flatten())
    rho_coords = np.append(rho_coords, radial_end_arm.flatten())
    theta_coords = np.append(angular_start_arm.flatten(), angular_transverse_points.flatten())
    theta_coords = np.append(theta_coords, angular_end_arm.flatten())
    return rho_coords, theta_coords


def generate_spiral_arm_densities(rho, transverse_densities_initial, h=h_default):
    """
    Args:
        rho: 1D array of radial distances to the spiral arm median
        transverse_densities_initial: 1D array of transverse distances from the spiral arm median. Initial simply indicates that these are the distances on one side of the spiral arm median
    Returns:
        density_spiral_arm: 1D array of densities for the spiral arm. The densities are calculated for the spiral arm median, the transverse points and the circular projection at the end points
        the densities appears in the same order as the spiral arm points in rho_coords and theta_coords as returned by generate_spiral_arm_points_spherical_coords
    """
    transverse_densities = np.append(1, transverse_densities_initial) # the 1 is to take into account the arm median itself
    transverse_densities = np.append(np.flip(transverse_densities_initial), transverse_densities) # so that we have the transverse densities on both sides of the spiral arm
    # calculate the densities for the arm median
    arm_median_densities = arm_median_density(rho, h) #1D array
    # calculate the transverse densities for the arm. Does not contain the contrbution from the circular projection at the end points, but has the arm median density
    # note the transverse densities is described by a Gausiian distribution, that's why the following works
    arm_transverse_densities = transverse_densities * arm_median_densities[:, np.newaxis] #2D array
    # calculate the densities for the end points projected in a circle around the end points
    density_start_arm = transverse_densities_initial * arm_median_densities[0] # this is a 1D array, but the same values goes for every index in start_arm along axis 0
    density_end_arm = transverse_densities_initial * arm_median_densities[-1] # this is a 1D array, but the same values goes for every index in end_arm along axis 0
    density_spiral_arm = np.concatenate([np.tile(density_start_arm, number_of_end_points), arm_transverse_densities.flatten(), np.tile(density_end_arm, number_of_end_points)])
    return density_spiral_arm
    

def plot_model_spiral_arm_densities(h=h_default, sigma_arm=sigma_arm_default):
    """
    Returns:
        a plot of the modelled spiral arm densities. Each arm is plotted in a separate subplot as a heatmap to indicate the density
    """
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(sigma_arm)
    # Define the number of rows and columns in the subplot grid
    num_rows, num_cols = 2, 2
    # Create subplots for each arm
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    # Adjust the horizontal and vertical spacing
    plt.subplots_adjust(hspace=0.01, wspace=0.3)
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            # generate the spiral arm medians
            theta, rho = spiral_arm_medians(arm_angles[i * num_cols + j], pitch_angles[i * num_cols + j])
            # generate the spiral arm points in spherical coordinates
            rho_coords, theta_coords = generate_spiral_arm_points_spherical_coords(rho, theta, pitch_angles[i * num_cols + j], transverse_distances)
            density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial, h)
            # Convert to Cartesian coordinates and plot the scatter plot
            x = rho_coords * np.cos(theta_coords)
            y = rho_coords * np.sin(theta_coords)
            ax.set_xlim(-40, 40)
            ax.set_ylim(-40, 40)
            ax.set_aspect('equal', adjustable='box')
            ax.scatter(x, y, c=density_spiral_arm, s=20)
            ax.set_xlabel('Distance in kpc from the Galactic center')
            ax.set_ylabel('Distance in kpc from the Galactic center')
            ax.set_title(f'Spiral Arm {spiral_arm_names[i * num_cols + j]}')
            ax.scatter(0, 0, c = 'magenta', s=10, label='Galactic centre')
            ax.scatter(0, r_s, c = 'gold', s=10, label='Sun')
            ax.legend()
    # Add a colorbar
    cbar = fig.colorbar(ax.collections[0], ax=axes, orientation='vertical')
    cbar.set_label('Density')
    plt.suptitle('Heatmap of the densities of spiral arms in our model')
    plt.savefig("output/spiral_arms_density_model.png", dpi=300)
    #plt.show()  # To display the plot


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
    cygnus_rho = rho_func(cygnus_long, cygnus_lat, cygnus_distance)
    cygnus_theta = theta_func(cygnus_long, cygnus_lat, cygnus_distance)
    cygnus_x = cygnus_rho * np.cos(cygnus_theta) 
    cygnus_y = cygnus_rho * np.sin(cygnus_theta)
    # Gum parameters
    gum_distance = 0.33 # kpc
    gum_long = np.radians(262)
    gum_lat = np.radians(0)
    gum_radius = 0.03 # kpc
    gum_rho = rho_func(gum_long, gum_lat, gum_distance)
    gum_theta = theta_func(gum_long, gum_lat, gum_distance)
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


def interpolate_density(gum_cygnus='False', method='cubic', h=h_default, sigma_arm=sigma_arm_default, arm_angles=arm_angles, pitch_angles=pitch_angles):
    """ Integrates the densities of the spiral arms over the entire galactic plane. The returned density is in units of kpc^-2. 
    Compared with the paper, it integrates P_\rho x P_\Delta at the top of page 6

    Args:
        grid_x (2D np.array): Contains all the x-values for the grid
        grid_y (2D np.array): Contaqins all the y-values for the grid
        method (str, optional): Interpolation method used in scipys griddata. Defaults to 'linear'.

    Returns:
        3D np.array: Interpolated densities for each spiral arm along axis 0. Axis 1 and 2 are the densities with respect to the grid
    """
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(sigma_arm) #d_min: minimum distance from the spiral arm
    #interpolated_densities = []
    x_grid = np.lib.format.open_memmap('output/galaxy_data/x_grid.npy')
    y_grid = np.lib.format.open_memmap('output/galaxy_data/y_grid.npy')
    #for i in range(len(arm_output_arm_medians(arm_angles[i], pitch_angles[i])
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i])
        # generate the spiral arm points in spherical coordinates
        rho_coords, theta_coords = generate_spiral_arm_points_spherical_coords(rho, theta, pitch_angles[i], transverse_distances)
        # generate the spiral arm densities
        density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial, h)
        # Convert to Cartesian coordinates
        x = rho_coords*np.cos(theta_coords)
        y = rho_coords*np.sin(theta_coords)
        # calculate interpolated density for the spiral arm
        interpolated_arm = griddata((x, y), density_spiral_arm, (x_grid, y_grid), method=method, fill_value=0)
        interpolated_arm[interpolated_arm < 0] = 0 # set all negative values to 0
        np.save(f'output/galaxy_data/interpolated_arm_{i}.npy', interpolated_arm)

        #interpolated_arm_sparse = csr_matrix(interpolated_arm)
        #interpolated_densities.append(interpolated_arm_sparse)
    if gum_cygnus == 'True':
        c_coords_x, c_coords_y, c_density, g_coords_x, g_coords_y, g_density = generate_gum_cygnus()
        interpolated_c = griddata((c_coords_x, c_coords_y), c_density, (x_grid, y_grid), method=method, fill_value=0)
        interpolated_g = griddata((g_coords_x, g_coords_y), g_density, (x_grid, y_grid), method=method, fill_value=0)
        #interpolated_densities.append(interpolated_c)
        #interpolated_densities.append(interpolated_g)
    #interpolated_densities = np.array(interpolated_densities)
    return #interpolated_densities


def plot_interpolated_galactic_densities(method='cubic', gum_cygnus = 'False', h=h_default, sigma_arm=sigma_arm_default, arm_angles=arm_angles, pitch_angles=pitch_angles):
    """
    Returns:
        a plot of the interpolated galactic densities. The galactic densities are interpolated from the individual spiral arm densities
        the plot is a heatmap to indicate the density, and all spiral arms are plotted in the same plot
    """
    grid_x, grid_y = np.mgrid[-20:20:1000j, -20:20:1000j]
    total_galactic_densities = interpolate_density(grid_x, grid_y, gum_cygnus, method, h, sigma_arm, arm_angles, pitch_angles)
    total_galactic_density = np.sum(total_galactic_densities, axis=0) # sum up all the arms
    #plot heatmap of the interpolated densities:
    plt.scatter(grid_x, grid_y, c=total_galactic_density.ravel(), cmap='viridis', s=1) # MISTAKE HERE: We are not really summing up all the latitudinal contribuitions, just plotting it on top of each other, overlapping
    plt.scatter(0, 0, c = 'magenta', s=2, label='Galactic centre')
    plt.scatter(0, r_s, c = 'gold', s=2, label='Sun')
    plt.gca().set_aspect('equal')
    plt.xlabel('Distance in kpc from the Galactic center')
    plt.ylabel('Distance in kpc from the Galactic center')
    plt.title('Heatmap of the interpolated densities of spiral arms in our model')
    plt.legend(loc='upper right')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    plt.savefig("output/interpolated_spiral_arms_density_model_5.png", dpi=1200)


def calc_effective_area_per_spiral_arm(method='cubic', h=h_default, sigma_arm=sigma_arm_default):
    """
    Calculates the effective area for each spiral arm. The density of each spiral arm is integrated over the entire galactic plane.
    The returned effective areas are in units of kpc^2, and appears in the same order as the spiral arms in arm_angles.
    """
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(sigma_arm) #d_min: minimum distance from the spiral arm
    d_x = 70 / 3000 # distance between each interpolated point in the x direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    d_y = 70 / 3000 # distance between each interpolated point in the y direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    grid_x, grid_y = np.mgrid[-35:35:3000j, -35:35:3000j]
    effective_area = []
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i])
        # generate the spiral arm points in spherical coordinates
        rho_coords, theta_coords = generate_spiral_arm_points_spherical_coords(rho, theta, pitch_angles[i], transverse_distances)
        # generate the spiral arm densities
        density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial, h)
        # in accordance with equation (9) in short.paper.2.0, the densities shall be scaled by their radial distance from the galactic centre
        # Convert to Cartesian coordinates
        x = rho_coords*np.cos(theta_coords)
        y = rho_coords*np.sin(theta_coords)
        # calculate interpolated density for the spiral arm
        interpolated_density = griddata((x, y), density_spiral_arm, (grid_x, grid_y), method, fill_value=0)
        interpolated_density[interpolated_density < 0] = 0 # set all negative values to 0
        
        # add the interpolated density to the total galactic density
        effective_area = np.append(effective_area, np.sum(interpolated_density) * d_x * d_y)
    filepath = "src/effective_area_per_spiral_arm.txt"
    np.savetxt(filepath, effective_area)
    return effective_area


def generate_latitudinal_points(b_max=5.0, db_above_1_deg = 0.2, b_min=0.01, b_lim_exp=1, scaling=0.015):
    """
    Args:
        b_min: minimum angular distance from the plane
        b_lim_exp: limit on angular distance from plane for the exponential distribution
        b_max: maximum angular distance from the plane
        scaling: scaling of the exponential distribution. A value of 0.015 generates between 110 - 130 points. Larger scaling = fewer points
    Returns:
        1D array with the angular distances from the Galactic plane, and an array with all increments (db) between the points
        Arrays made ready to be used in a central Rieman sum 
    """
    # old
    """ db = 0.2   # increments in db (degrees):
    latitudes = np.radians(np.arange(-5, 5 + db, db))
    db = np.ones(num_lats) * db
    db = np.radians(db)
    return latitudes, db """
    
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
            np.save('output/galaxy_data/latitudes.npy', np.radians(latitudes)) # saving latitudes
            np.save('output/galaxy_data/db.npy', np.radians(db)) # saving db
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
            np.save('output/galaxy_data/latitudes.npy', np.radians(latitudes)) # saving latitudes
            np.save('output/galaxy_data/db.npy', np.radians(db)) # saving db
            return 
        else:
            raise ValueError("The number of latitudinal angles is not an integer. Please change the value of db_above_1_deg")
    else:
        print("The first element of transverse_distances is not b_min. Trying again...")
        return generate_latitudinal_points(b_max, db_above_1_deg, b_min, b_lim_exp, scaling)
    

def calculate_galactic_coordinates(b_max=1, db_above_1_deg = 0.1, b_min=0.01, b_lim_exp=1, scaling=0.015):
    logging.info("Calculating the galactic coordinates")
    # Calculate coordinates
    dr = 0.01   # increments in dr (kpc):
    dl = 0.2   # increments in dl (degrees):
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    dl = np.radians(dl)
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(dr, r_s + rho_max + 5 + dr, dr) # +5 kpc to take into account the width of the spiral arms
    # save coordinates to disk
    np.save('output/galaxy_data/longitudes.npy', longitudes) # saving longitudes
    np.save('output/galaxy_data/radial_distances.npy', radial_distances) # saving radial_distances
    np.save('output/galaxy_data/dr.npy', dr) # saving dr
    np.save('output/galaxy_data/dl.npy', dl) # saving dl
    generate_latitudinal_points(b_max, db_above_1_deg, b_min, b_lim_exp, scaling)
    latitudes = np.load('output/galaxy_data/latitudes.npy')
    logging.info("Length of latitudes: " + str(len(latitudes)) + ". Length of radial_distances: " + str(len(radial_distances)) + ". Length of longitudes: " + str(len(longitudes)) + ".")
    logging.info("Max latitude: " + str(np.degrees(latitudes[-1])) + ". Min latitude: " + str(np.degrees(latitudes[0])) + ".")
    logging.info("Creating coordinate grids")
    radial_grid, long_grid, lat_grid = np.meshgrid(radial_distances, longitudes, latitudes, indexing='ij')
    np.save('output/galaxy_data/radial_grid.npy', radial_grid.ravel()) # saving radial_grid
    np.save('output/galaxy_data/long_grid.npy', long_grid.ravel()) # saving long_grid
    np.save('output/galaxy_data/lat_grid.npy', lat_grid.ravel()) # saving lat_grid
    del radial_grid, long_grid, lat_grid, latitudes, longitudes, radial_distances, dr, dl
    gc.collect()
    logging.info("Grids created. Now calculating the latidudinal cosinus")
    latitudinal_cosinus = np.cos(np.lib.format.open_memmap('output/galaxy_data/lat_grid.npy'))
    np.save('output/galaxy_data/latitudinal_cosinus.npy', latitudinal_cosinus) # saving latitudinal_cosinus
    del latitudinal_cosinus
    gc.collect()
    logging.info("Latitudinal cosinus calculated. Now calculating z_grid and height_distribution_values")
    z_grid = np.lib.format.open_memmap('output/galaxy_data/radial_grid.npy') * np.sin(np.lib.format.open_memmap('output/galaxy_data/lat_grid.npy'))
    np.save('output/galaxy_data/z_grid.npy', z_grid) # saving z_grid
    del z_grid
    gc.collect()
    height_distribution_values = height_distribution(np.lib.format.open_memmap('output/galaxy_data/z_grid.npy'))
    np.save('output/galaxy_data/height_distribution_values.npy', height_distribution_values) # saving height_distribution_values
    del height_distribution_values
    gc.collect()
    logging.info("z_grid and height_distribution_values calculated. Now calculating rho, theta, x and y coordinates")
    rho_coords_galaxy = rho_func(np.lib.format.open_memmap('output/galaxy_data/long_grid.npy'), np.lib.format.open_memmap('output/galaxy_data/lat_grid.npy'), np.lib.format.open_memmap('output/galaxy_data/radial_grid.npy'))
    np.save('output/galaxy_data/rho_coords_galaxy.npy', rho_coords_galaxy) # saving rho_coords_galaxy
    del rho_coords_galaxy
    gc.collect()
    logging.info("rho coordinates calculated")
    theta_coords_galaxy = theta_func(np.lib.format.open_memmap('output/galaxy_data/long_grid.npy'), np.lib.format.open_memmap('output/galaxy_data/lat_grid.npy'), np.lib.format.open_memmap('output/galaxy_data/radial_grid.npy'))
    np.save('output/galaxy_data/theta_coords_galaxy.npy', theta_coords_galaxy) # saving theta_coords_galaxy
    del theta_coords_galaxy
    gc.collect()
    logging.info("theta coordinates calculated. Now removing the grid files from disk")
    os.remove('output/galaxy_data/radial_grid.npy')
    os.remove('output/galaxy_data/long_grid.npy')
    os.remove('output/galaxy_data/lat_grid.npy')
    logging.info("Calculating x-values")
    x_grid = np.lib.format.open_memmap('output/galaxy_data/rho_coords_galaxy.npy') * np.cos(np.lib.format.open_memmap('output/galaxy_data/theta_coords_galaxy.npy'))
    np.save('output/galaxy_data/x_grid.npy', x_grid) # saving x_grid
    del x_grid
    gc.collect()
    logging.info("Calculating y-values")
    y_grid = np.lib.format.open_memmap('output/galaxy_data/rho_coords_galaxy.npy') * np.sin(np.lib.format.open_memmap('output/galaxy_data/theta_coords_galaxy.npy'))
    np.save('output/galaxy_data/y_grid.npy', y_grid) # saving y_grid
    del y_grid
    gc.collect()
    # delte from disk the rho_coords_galaxy, theta_coords_galaxy:
    logging.info("x and y coordinates calculated. Now removing the rho and theta coordinates from disk")
    os.remove('output/galaxy_data/rho_coords_galaxy.npy')
    os.remove('output/galaxy_data/theta_coords_galaxy.npy')
    return
    

def calc_modelled_emissivity(b_max=1, db_above_1_deg = 0.1, fractional_contribution=fractional_contribution_default, gum_cygnus='False', method='linear', readfile=True, h=h_default, sigma_arm=sigma_arm_default, arm_angles=arm_angles, pitch_angles=pitch_angles):
    logging.info("Calculating modelled emissivity of the Milky Way")
    if readfile == True:
        effective_area = np.loadtxt("src/effective_area_per_spiral_arm.txt")
    elif readfile == False:
        effective_area = calc_effective_area_per_spiral_arm(method, h, sigma_arm)
    else:
        raise ValueError("readfile must be either True or False")
        
    calculate_galactic_coordinates(b_max, db_above_1_deg)
    num_lats = len(np.lib.format.open_memmap('output/galaxy_data/latitudes.npy'))
    num_rads = len(np.lib.format.open_memmap('output/galaxy_data/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap('output/galaxy_data/longitudes.npy'))
    logging.info("Coordinates calculated and read from disk. Now interpolating each spiral arm")
    # coordinates made. Now we need to interpolate each spiral arm and sum up the densities
    interpolate_density(gum_cygnus, method, h, sigma_arm, arm_angles, pitch_angles)
    logging.info("Density calculated. Now calculating the emissivity")
    common_multiplication_factor = total_galactic_n_luminosity * np.lib.format.open_memmap('output/galaxy_data/height_distribution_values.npy')
    emissivity_rad_long_lat = np.zeros((num_rads, num_longs, num_lats)) # to store the intensity as a function of radius, longitude and latitude. Used for MC-simulation of the galaxy
    for i in range(4):
        logging.info(f"Calculating spiral arm number: {i+1}")
        scaled_arm_emissivity = np.load(f'output/galaxy_data/interpolated_arm_{i}.npy') * common_multiplication_factor * fractional_contribution[i] / (effective_area[i] * kpc**2) # spiral arms
        # save this scaled density 
        np.save(f'output/galaxy_data/interpolated_arm_emissivity_{i}.npy', scaled_arm_emissivity)
        # reshape this 1D array into 3D array to facilitate for the summation over the different longitudes and also the MonteCarlo Simulation
        scaled_arm_emissivity = scaled_arm_emissivity.reshape((num_rads, num_longs, num_lats))
        emissivity_rad_long_lat += scaled_arm_emissivity
    # Following files to be used for MC-simulation of the galaxy
    np.save('output/galaxy_data/emissivity_longitudinal.npy', np.sum(emissivity_rad_long_lat, axis=(0, 2))) # Sum up all emissivities for all LOS for each value of longitude. Without running average.
    np.save('output/galaxy_data/emissivity_long_lat.npy', np.sum(emissivity_rad_long_lat, axis=(0))) # sum over all radii to get a map for emissivity in the long, lat plane.
    np.save('output/galaxy_data/emissivity_rad_long_lat.npy', emissivity_rad_long_lat) # store the entire 3D array of emissivity.
    return  


def calc_modelled_intensity(b_max=1, db_above_1_deg = 0.1, fractional_contribution=fractional_contribution_default, gum_cygnus='False', method='linear', readfile=True, h=h_default, sigma_arm=sigma_arm_default, arm_angles=arm_angles, pitch_angles=pitch_angles):
    logging.info("Calculating the modelled NII intensity of the Milky Way")
    calc_modelled_emissivity(b_max, db_above_1_deg, fractional_contribution, gum_cygnus, method, readfile, h, sigma_arm, arm_angles, pitch_angles)
    num_lats = len(np.lib.format.open_memmap('output/galaxy_data/latitudes.npy'))
    num_rads = len(np.lib.format.open_memmap('output/galaxy_data/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap('output/galaxy_data/longitudes.npy'))
    db = np.load('output/galaxy_data/db.npy')
    dr = np.load('output/galaxy_data/dr.npy')
    dl = np.load('output/galaxy_data/dl.npy')
    
    latitudinal_cosinus = np.lib.format.open_memmap('output/galaxy_data/latitudinal_cosinus.npy')
    common_multiplication_factor =  dr * latitudinal_cosinus/ (4 * np.pi * np.radians(b_max * 2) * np.radians(5))
    common_multiplication_factor = np.reshape(common_multiplication_factor, (num_rads, num_longs, num_lats)) * db[np.newaxis, np.newaxis, :] # reshaping to facilitate the multiplication with non-uniform latitudinal increments db
    common_multiplication_factor = common_multiplication_factor.ravel() #unraveling so that we can multiply with the interpolated densities
    del latitudinal_cosinus
    gc.collect()
    intensities_per_arm = np.zeros((4, num_longs)) # to store the intensity as a function of longitude for each spiral arm. Used for the intesnity-plots to compare with Higdon & Lingenfelter
    for i in range(4):
        logging.info(f"Calculating spiral arm number: {i+1}")
        """ if i==4:
            interpolated_densities[i] *= cygnus_nii_luminosity * db * dr * latitudinal_cosinus / ((4 * np.pi) * kpc**2) # cygnus
        elif i==5:
            interpolated_densities[i] *= gum_nii_luminosity * db * dr * latitudinal_cosinus / ((4 * np.pi) * kpc**2) # gum
        else: """
        arm_intensity = np.load(f'output/galaxy_data/interpolated_arm_emissivity_{i}.npy') * common_multiplication_factor # spiral arms
        # reshape this 1D array into 3D array to facilitate for the summation over the different longitudes
        arm_intensity = arm_intensity.reshape((num_rads, num_longs, num_lats))
        # sum up to get the intensity as a function of longitude
        arm_intensity = arm_intensity.sum(axis=(0, 2)) # sum up all densities for all LOS for each value of longitude
        window_size = 5 / np.degrees(dl) # 5 degrees in divided by the increment in degrees for the longitude. This is the window size for the running average, number of points
        arm_intensity = running_average(arm_intensity, window_size) /window_size # running average to smooth out the density distribution
        intensities_per_arm[i] += arm_intensity
    b_filename = str(b_max).replace(".", "_")
    filename_intensity_data = f'output/galaxy_data/intensities_per_arm_b_max_{b_filename}.npy'
    np.save(filename_intensity_data, intensities_per_arm) # saving the same array we are plotting usually. Sum over all spiral arms to get one longitudinal map. With running average
    return


def plot_modelled_intensity_per_arm(filename_output = "output/modelled_intensity.png", filename_intensity_data = 'output/galaxy_data/intensities_per_arm.npy', fractional_contribution=fractional_contribution_default, gum_cygnus='False',h=h_default, sigma_arm=sigma_arm_default):
    """
    Plots the modelled intensity of the Galactic disk as a function of Galactic longitude.
    """
    # plot the FIRAS data for the NII 205 micron line
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # plot the modelled intensity
    intensities_per_arm = np.lib.format.open_memmap(filename_intensity_data)
    longitudes = np.lib.format.open_memmap('output/galaxy_data/longitudes.npy')
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[0], label=f"NC. f={fractional_contribution[0]}")
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[1], label=f"P. $\ $ f={fractional_contribution[1]}")
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[2], label=f"SA. f={fractional_contribution[2]}")
    plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[3], label=f"SC. f={fractional_contribution[3]}")
    if gum_cygnus == 'True': 
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[4], label="Cy.")
        plt.plot(np.linspace(0, 360, len(longitudes)), intensities_per_arm[5], label="Gu.")
    plt.plot(np.linspace(0, 360, len(longitudes)), np.sum(intensities_per_arm, axis=0), label="Total")
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
    plt.text(0.02, 0.9, fr'NII Luminosity = {total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig(filename_output, dpi=1200)
    plt.close()
    #plt.show()


def plot_modelled_emissivity_total(fractional_contribution, gum_cygnus='False', method='linear', readfile = "true", filename = "output/modelled_emissivity.png", h=h_default, sigma_arm=sigma_arm_default):
    """
    Plots the modelled emissivity of the Galactic disk as a function of Galactic longitude.
    """
    longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, gum_cygnus, method, readfile, h, sigma_arm)
    print("interpolated densities: ", densities_as_func_of_long.shape)
    plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
    print(np.sum(densities_as_func_of_long))
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    plt.gca().set_aspect('equal')
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig(filename, dpi=1200)
    plt.show()


def test_interpolation_method_interpolated_densities(h=2.4):
    methods = ['linear', 'nearest', 'cubic']
    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.6)
    for i in range(len(methods)):
        grid_x, grid_y = np.mgrid[-20:20:1000j, -20:20:1000j]
        total_galactic_densities = interpolate_density(grid_x, grid_y, methods[i], h)
        total_galactic_density = np.sum(total_galactic_densities, axis=0)
        # heatmap of the interpolated densities:
        ax = axes[i]
        ax.set_aspect('equal', adjustable='box')
        scatter = ax.scatter(grid_x, grid_y, c=total_galactic_density.flatten(), cmap='viridis', s=20)
        ax.set_xlabel('Distance in kpc from the Galactic center')
        ax.set_ylabel('Distance in kpc from the Galactic center')
        ax.set_title(f'Interpolated densities using the {methods[i]} method')
        ax.scatter(0, 0, c = 'magenta', s=2, label='Galactic centre')
        ax.scatter(0, r_s, c = 'gold', s=2, label='Sun')
        ax.legend()
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
        cbar.set_label('Density')        
    print("Done with plotting")
    plt.suptitle('Testing of interpolation methods')
    print("Saving figure...")
    plt.savefig("output/test_interpolation_methods_interpolated_densities.png", dpi=1200, bbox_inches='tight')
    print("Done saving figure")
    plt.show()  # To display the plot
    

def test_fractional_contribution(method='linear', readfile='true', h=h_default, sigma_arm=sigma_arm_default):
    num_rows, num_cols = 2, 2
    # array with fractional contribution of each spiral arm to be testet. Respectively respectively Norma-Cygnus(NC), Perseus(P), Sagittarius-Carina(SA), Scutum-Crux(SC)
    fractional_contribution = [[0.25, 0.25, 0.25, 0.25],
                               [0.18, 0.36, 0.18, 0.28],
                               [0.15, 0.39, 0.15, 0.31],
                               [0.17, 0.34, 0.15, 0.34]]
    # Create subplots for each arm
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    # Adjust the horizontal and vertical spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.12)
    # wspace=0.6 became too tight
    for i in range(num_rows):
        for j in range(num_cols):
            print("Calculating with fractional contribution list: ", i * num_cols + j + 1)
            ax = axes[i, j]
            longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution[i * num_cols + j], method, readfile)
            print(longitudes.shape, densities_as_func_of_long.shape, np.sum(densities_as_func_of_long, axis=0).shape)
            print(np.linspace(0, 100, len(longitudes)))
            print(np.sum(densities_as_func_of_long, axis=0))
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[i * num_cols + j][0]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[i * num_cols + j][1]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[i * num_cols + j][2]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[i * num_cols + j][3]}")
            ax.plot(np.linspace(0, 360, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
            ax.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=ax.transAxes, fontsize=8, color='black')
            # Redefine the x-axis labels to match the values in longitudes
            x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
            ax.set_xticks(x_ticks) #np.linspace(0, 360, 13), 
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel("Galactic longitude l (degrees)")
            ax.set_ylabel("Modelled emissivity")
            ax.set_title("Modelled emissivity of the Galactic disk")
            ax.legend()         
    print("Done with plotting. Saving figure...") 
    plt.suptitle('Testing different values for the fractional contribution of each spiral arm')
    plt.savefig("output/test_fractional_contribution2", dpi=1200, bbox_inches='tight')
    #plt.show()  # To display the plot


def test_disk_scale_length(method='linear', readfile='true', fractional_contribution=fractional_contribution_default, sigma_arm=sigma_arm_default):
    # Function to test different values for the disk scale length to see which gives the best fit compared to the data
    disk_scale_lengths = np.array([1.8, 2.1, 2.4, 2.7, 3.0])
    linestyles = np.array(['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10))])
    for i in range(len(disk_scale_lengths)):
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, method, readfile, h=disk_scale_lengths[i], sigma_arm=sigma_arm)
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), linestyile=linestyles[i], color='black', label=f"$H_\rho$ = {disk_scale_lengths[i]} kpc")
        # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    plt.gca().set_aspect('equal')
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig('output/test_disk_scale_length', dpi=1200)
    plt.show()

def test_transverse_scale_length(method='linear', readfile='true', fractional_contribution=fractional_contribution_default, h=sigma_arm_default):
    transverse_scale_lengths = np.array([0.25, 0.4, 0.5, 0.6, 0.75])
    linestyles = np.array(['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10))])
    for i in range(len(transverse_scale_lengths)):
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, method, readfile, h=h, sigma_arm=transverse_scale_lengths[i])
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), linestyile=linestyles[i], color='black', label=f"$\sigma_A$ = {transverse_scale_lengths[i]} kpc")
        # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    plt.gca().set_aspect('equal')
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig('output/transverse_scale_length', dpi=1200)
    plt.show()


def find_max_value_and_index(arr):
    if not arr.any():
        return None, None  # Return None if the array is empty

    max_value = max(arr)
    max_index = np.argmax(arr)

    return max_value, max_index


def find_arm_tangents(fractional_contribution=fractional_contribution_default, gum_cygnus='False', method='cubic', readfile = "false", filename = "output/test_arm_angles/test_arm_start_angle.txt", h=h_default, sigma_arm=sigma_arm_default):
    # starting angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
    # arm_angles = np.radians([70, 160, 250, 340]) #original
    nc_angle = np.arange(60, 81, 1)
    p_angle = np.arange(150, 171, 1)
    sa_angle = np.arange(240, 261, 1)
    sc_angle = np.arange(330, 351, 1)
    for i in range(len(nc_angle)):
        angles = np.radians(np.array([nc_angle[i], p_angle[i], sa_angle[i], sc_angle[i]]))
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, gum_cygnus, method, readfile, h, sigma_arm, angles)        
        _, max_index_nc = find_max_value_and_index(densities_as_func_of_long[0])
        _, max_index_p = find_max_value_and_index(densities_as_func_of_long[1])
        _, max_index_sa = find_max_value_and_index(densities_as_func_of_long[2])
        _, max_index_sc = find_max_value_and_index(densities_as_func_of_long[3])
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[0]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[1]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[2]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[3]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
        print(np.sum(densities_as_func_of_long))
        # Redefine the x-axis labels to match the values in longitudes
        x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
        plt.xticks(np.linspace(0, 100, 13), x_ticks)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("Galactic longitude l (degrees)")
        plt.ylabel("Modelled emissivity")
        plt.title("Modelled emissivity of the Galactic disk")
        # Add parameter values as text labels
        plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.9, fr'NII Luminosity = {total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.85, fr'Arm angles: nc={nc_angle[i]}, p={p_angle[i]}, sa={sa_angle[i]}, sc={sc_angle[i]}', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.legend()
        plt.savefig(f'output/test_arm_angles/set_{i}', dpi=1200)
        plt.close()
        # save to file filename
        with open(filename, 'a') as f:
            f.write(f"{nc_angle[i]} {p_angle[i]} {sa_angle[i]} {sc_angle[i]} {np.degrees(longitudes[max_index_nc])} {np.degrees(longitudes[max_index_p])} {np.degrees(longitudes[max_index_sa])} {np.degrees(longitudes[max_index_sc])}\n")
        
        
def find_pitch_angles(fractional_contribution=fractional_contribution_default, gum_cygnus='False', method='cubic', readfile = "false", filename = "output/test_pitch_angles_4/test_pitch_angles_4.txt", h=h_default, sigma_arm=sigma_arm_default):
    # starting angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
    # arm_angles = np.radians([70, 160, 250, 340]) #original
    # pitch_angles = np.radians(np.array([13.5, 13.5, 13.5, 15.5])) # original
    # Vallées pitch angles: 12.8
    
    #pitch_angles = np.arange(12, 15.6, 0.1)
    pitch_angles = np.arange(13.5, 15.6, 0.1)
    #Arm_Angles = np.array([65, 160, 245, 335]) #1
    #Arm_Angles = np.array([65, 155, 240, 330]) #2
    #Arm_Angles = np.array([65, 160, 250, 330]) #3
    Arm_Angles = np.array([65, 160, 240, 330]) #4
    for i in range(len(pitch_angles)):
        Pitch_Angles = np.radians([pitch_angles[i], pitch_angles[i], pitch_angles[i], pitch_angles[i]])
        longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, gum_cygnus, method, readfile, h, sigma_arm, np.radians(Arm_Angles), Pitch_Angles)        
        _, max_index_nc = find_max_value_and_index(densities_as_func_of_long[0])
        _, max_index_p = find_max_value_and_index(densities_as_func_of_long[1])
        _, max_index_sa = find_max_value_and_index(densities_as_func_of_long[2])
        _, max_index_sc = find_max_value_and_index(densities_as_func_of_long[3])
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[0]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[1]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[2]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[3]}")
        plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
        print(np.sum(densities_as_func_of_long))
        # Redefine the x-axis labels to match the values in longitudes
        x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
        plt.xticks(np.linspace(0, 100, 13), x_ticks)
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel("Galactic longitude l (degrees)")
        plt.ylabel("Modelled emissivity")
        plt.title("Modelled emissivity of the Galactic disk")
        # Add parameter values as text labels
        plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.9, fr'NII Luminosity = {total_galactic_n_luminosity:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.85, fr'Arm angles: nc={Arm_Angles[0]}, p={Arm_Angles[1]}, sa={Arm_Angles[2]}, sc={Arm_Angles[3]}', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.text(0.02, 0.8, fr'Pitch angles = {pitch_angles[i]}', transform=plt.gca().transAxes, fontsize=8, color='black')
        plt.legend()
        plt.savefig(f'output/test_pitch_angles_4/set_{i}', dpi=1200)
        plt.close()
        # save to file filename
        with open(filename, 'a') as f:
            f.write(f"{pitch_angles[i]} {np.degrees(longitudes[max_index_nc])} {np.degrees(longitudes[max_index_p])} {np.degrees(longitudes[max_index_sa])} {np.degrees(longitudes[max_index_sc])}\n")


def plot_from_file():
    x_grid = np.load('output/galaxy_data/x_grid.npy')
    y_grid = np.load('output/galaxy_data/y_grid.npy')
    z_grid = np.load('output/galaxy_data/z_grid.npy')
    total_galactic_density_unweighted = np.load('output/galaxy_data/total_galactic_density_unweighted.npy')
    total_galactic_density_weighted = np.load('output/galaxy_data/total_galactic_density_weighted.npy')
    print("total_galactic_density_weighted: ", total_galactic_density_weighted.shape)
    print("total_galactic_density_unweighted: ", total_galactic_density_unweighted.shape)
    


def calc_and_plot():
    #calc_modelled_intensity() # calculates coordinates, emissivity and intensity. Writes data to file
    #test_plot_density_distribution()
    a = 12


def test_max_b():
    #b_max = np.array([0.5, 1, 3.5, 5])
    #db_above_1_deg = np.array([0, 0, 0.1, 0.2])
    b_max = np.array([5])
    db_above_1_deg = np.array([0.2])
    for i in range(len(b_max)):
        calc_modelled_intensity(b_max=b_max[i], db_above_1_deg=db_above_1_deg[i])
        b_filename = str(b_max[i]).replace(".", "_")
        filename_output = f"output/modelled_intensity_b_max_{b_filename}.png"
        filename_intensity_data = f'output/galaxy_data/intensities_per_arm_b_max_{b_filename}.npy'
        plot_modelled_intensity_per_arm(filename_output, filename_intensity_data)

def main() -> None:
    # other levels for future reference: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    logging.basicConfig(level=logging.INFO) 
    #test_max_b()
    
    #calc_and_plot()

if __name__ == "__main__":
    main()


#plot_from_file() 

# consider this: for every point in the galactic plane, we have a fixed number of points in the z-direction: 21. How are these distributed?
# For the xy points closest to the earth, the z-points are distributed very close to the galactic plane. For the xy points furthest away from the earth, the z-points are distributed further away from the galactic plane.
# Thus, when we add the transverse contributions together, we get a very skewed distribution of the z-points.




#find_arm_tangents()
#find_pitch_angles()

# SO the pitch angles must also be changed. 
# For SA: the larger starting angle, the closer to GC the luminocity is concentrated. Will thus adopt a starting angle of 245 degrees for next test run
# For SC: the larger starting angle, the closer to GC the luminocity is concentrated. Will thus adopt a starting angle of 335 degrees for next test run
# For P: the smaller the starting angle, the closer to GC the luminocity is concentrated. Also, the height of the peak is reduced by about 20% between the smallest and largest starting angle. Will keep the starting angle at 160 degrees for next test run
# For NC: the higher the value, the more right-skewed the luminocity distribution becomes. Will set the starting angle at 65 degrees for next test run. Also, a larger angle makes the "dump" in the middle les dumpy, and vice versa.
#plot_interpolated_galactic_densities()
#plot_spiral_arms()
#plot_model_spiral_arm_densities()
#calc_effective_area_per_spiral_arm()
#['Norma-Cygnus', 'Perseus', 'Sagittarius-Carina', 'Scutum-Crux']
fractional_contribution = [0.17, 0.30, 0.22, 0.31] # fractional contribution of each spiral arm to the total NII 205 micron line intensity
#plot_modelled_emissivity_per_arm(fractional_contribution, 'linear', 'true', "output/modelled_emissivity_17_34_15_34.png")
#test_fractional_contribution()
#test_interpolation_method()
#c_coords, g_coords, c_density, g_density = generate_gum_cygnus()

Arm_Angles = np.radians([65, 160, 245, 335])
#calc_modelled_emissivity(fractional_contribution=fractional_contribution_default, gum_cygnus='False', sigma_arm=False, method='cubic', readfile='false')
#plot_interpolated_galactic_densities() 
#test_interpolation_method_interpolated_densities()
#print("long_lat_rad_coords_generation")
#test_long_lat_rad_coords_generation()
#test_spherical_interpolation(fractional_contribution, 'cubic', 'true')
#calc_modelled_emissivity(fractional_contribution, 'cubic', 'true')

# plot number 18: 220M points. Plot number 17: 25M points. 
#test_fractional_contribution()

#c_coords, g_coords, c_density, g_density = generate_gum_cygnus()

# modelled_emissivity_arms:
# 4: /np.radians(1), db = 1
# 5: /np.radians(1) and height_distribution_values = 1, db = 1
# 6: /np.radians(7) and latitudes |b| < 3.5 degrees, db = 1
# 7: /np.radians(7) and latitudes |b| < 3.5 degrees and /dl, db = 1
# 8: /np.radians(1) and latitudes |b| < 0.5 degrees and /dl, db = 1
# 9: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 1
# 10: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 0.1
# 11: /np.radians(7) and latitudes |b| < 3.5 degrees, db = 0.5, [0.17, 0.34, 0.15, 0.34]
# 12: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 0.5
# 13: /np.radians(1) and latitudes |b| < 0.5 degrees and /dl, db = 0.1
# 14: /np.radians(7) and latitudes |b| < 3.5 degrees, db = 0.5, [0.17, 0.38, 0.15, 0.30]
# 15: /np.radians(7) and latitudes |b| < 3.5 degrees, db = 0.5, [0.18, 0.36, 0.18, 0.28] ########################################
# 16: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 0.1, [0.18, 0.36, 0.18, 0.28]
# 17: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 0.1, [0.18, 0.33, 0.21, 0.28]
# 18: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 0.1, [0.18, 0.31, 0.21, 0.30]
# 19: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 0.1, [0.18, 0.30, 0.21, 0.31]
# 20: /np.radians(1) and latitudes |b| < 0.5 degrees, db = 0.1, [0.17, 0.30, 0.22, 0.31]

# modelled_emissivity_arms_running_average_7degree5.png: increased dl from .2 to .5. Effect: made the intensity higher in value, but the shape is the same
# modelled_emissivity_arms_running_average_7degree6.png: dl = 0.1. No visible improvement
# modelled_emissivity_arms_running_average_7degree7.png: dl = 0.2. Remove start-points. # somehow increased the modelled intensity, and the spiral arm contributions become wrong
# modelled_emissivity_arms_running_average_7degree8.png: dl = 0.2. Increase number of end-points from 181 to 300. Did not make any visible difference
# modelled_emissivity_arms_running_average_7degree9.png: dl = 0.2. Increase number of spiral-arm-medians
# modelled_emissivity_arms_running_average_7degree10.png: dl = 0.2. Increase number of transverse points
#  NOTHING THUSS FAR HAVE IMPROVED THE MODEL

#  REMEMBER THE LUMINOSITY IS SIMPLY MULTIPLIED IN. SO, IF MY MODEL PRODUCES TOO HIGH VALUES, MY FINAL MODELLED INTENSITY WILL TOO BECOME TOO HIGH! thus I could try to decrease the number of points I use
# modelled_emissivity_arms_running_average_7degree11.png: dl = 0.2. Changed scaling from 0.03 to 0.5, i.e. decrease the number of transverse points
# modelled_emissivity_arms_running_average_7degree12.png: dl = 0.2. Changed scaling from 0.03 to 0.1, i.e. decrease the number of transverse points
# modelled_emissivity_arms_running_average_7degree13.png: dl = 0.2. Changed scaling from 0.03 to 0.2, i.e. decrease the number of transverse points
# modelled_emissivity_arms_running_average_7degree14.png: dl = 0.2. Changed scaling from 0.03 to 0.3, i.e. decrease the number of transverse points
# modelled_emissivity_arms_running_average_7degree15.png: dl = 0.2. Changed scaling from 0.03 to 0.2, i.e. decrease the number of transverse points. Also decreased the number of end points to 45
# modelled_emissivity_arms_running_average_7degree16.png: dl = 0.2. Changed scaling from 0.03 to 0.2, i.e. decrease the number of transverse points. Also decreased the number of end points to 45. dtheta from 0.01 to 0.1
# modelled_emissivity_arms_running_average_7degree17.png: dl = 0.2. Changed scaling from 0.03 to 0.2, i.e. decrease the number of transverse points. Also decreased the number of end points to 20.

# Changed scaling from 0.03 to 0.2, i.e. decrease the number of transverse points. Also decreased the number of end points to 45. Had no visible effect

# modelled_emissivity_arms_running_average_7degree18.png: changed r_s to 8.5 kpc
# modelled_emissivity_arms_running_average_7degree19.png: changed r_s to 8.178 kpc
# modelled_emissivity_arms_running_average_7degree20.png: Changed rho_max to 10 kpc
# modelled_emissivity_arms_running_average_7degree21.png: Changed rho_max to 10 kpc, rho_mihn to 3kpc
# modelled_emissivity_arms_running_average_7degree22.png: Changed rho_max to 1.3*r_s kpc, rho_mihn to 0.39*r_s
# modelled_emissivity_arms_running_average_7degree23.png: Changed rho_max to 10 kpc, rho_mihn to 3kpc. r_s = 8.178 kpc
# modelled_emissivity_arms_running_average_7degree24.png: r_s = 8.178 kpc. |b| < 5 degrees, db = 0.5, r_max = 10
# modelled_emissivity_arms_running_average_7degree25.png: r_s = 8.178 kpc. |b| < 5 degrees, db = 0.3, r_max = 10
# modelled_emissivity_arms_running_average_7degree26.png: r_s = 8.178 kpc. |b| < 5 degrees, db = 0.2, r_max = 10. Tiny improvement over 24
# modelled_emissivity_arms_running_average_7degree27.png: r_s = 7.6 kpc. |b| < 5 degrees, db = 0.2, r_max = 10
# modelled_emissivity_arms_running_average_7degree28.png: r_s = 7.6 kpc. |b| < 5 degrees, db = 0.5, r_max = 35
# modelled_emissivity_arms_running_average_7degree29.png: r_s = 8.178 kpc. |b| < 5 degrees, db = 0.5, r_max = 35
# modelled_emissivity_arms_running_average_7degree30.png: r_s = 8.178 kpc. |b| < 5 degrees, db = 0.5, r_max = 35, changed arm angles
# modelled_emissivity_arms_running_average_7degree31.png: r_s = 8.178 kpc. |b| < 5 degrees, db = 0.5, r_max = 35, changed arm angles
# modelled_emissivity_arms_running_average_7degree32.png: r_s = 8.178 kpc. |b| < 5 degrees, db = 0.5, r_max = 35, changed arm angles
# modelled_emissivity_arms_running_average_7degree33.png: same as 32, checking if there is some bugs here explaining the weird results from find_arm_tangents
# modelled_emissivity_arms_running_average_7degree34.png: same as 32, used the current optimal arm angles from first run of find_arm_tangents
# modelled_emissivity_arms_running_average_7degree35.png: same as 32, testing gum_cycgnus contribution
# modelled_emissivity_arms_running_average_7degree36.png: same as 32, testing gum_cycgnus contribution. Removing running average
# modelled_emissivity_arms_running_average_7degree37.png: same as 32, testing gum_cycgnus contribution. Removing running average and /np.rad(10). Do not average the gaussian


ARM_ANGLES1 = np.radians([65, 160, 250, 330])
ARM_ANGLES2 = np.radians([65, 160, 245, 330])
ARM_ANGLES3 = np.radians([65, 160, 240, 330])
PITCH_ANGLES_1 = np.radians([15.5, 13.5, 15.5, 16.3])
PITCH_ANGLES_2 = np.radians([15.5, 15.5, 15.5, 16.3])
PITCH_ANGLES_3 = np.radians([15.5, 14, 15, 16.3])
PITCH_ANGLES_4 = np.radians([15.5, 14, 14.5, 16.3])
PITCH_ANGLES_5 = np.radians([15.5, 14, 14, 16.3])
PITCH_ANGLES_6 = np.radians([14, 14, 14, 16.3])
PITCH_ANGLES_7 = np.radians([14, 14, 14, 16])
# modelled_emissivity_arms_running_average_7degree38.png: same as 32, testing final arm angles and pitch angles1. 
# modelled_emissivity_arms_running_average_7degree39.png: same as 32, testing final arm angles and pitch angles2. 

#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree38.png", h_default, sigma_arm_default, ARM_ANGLES1, PITCH_ANGLES_1)
#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree39.png", h_default, sigma_arm_default, ARM_ANGLES1, PITCH_ANGLES_2)
#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree40.png", h_default, sigma_arm_default, ARM_ANGLES2, PITCH_ANGLES_3)
#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree41.png", h_default, sigma_arm_default, ARM_ANGLES2, PITCH_ANGLES_4)
#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree42.png", h_default, sigma_arm_default, ARM_ANGLES3, PITCH_ANGLES_5)
#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree43.png", h_default, sigma_arm_default, ARM_ANGLES3, PITCH_ANGLES_6)
#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree44.png", h_default, sigma_arm_default, ARM_ANGLES3, PITCH_ANGLES_7)
#plot_modelled_emissivity_per_arm([0.18, 0.36, 0.18, 0.28], 'False', False,'cubic', 'false', "output/modelled_emissivity_arms_running_average_7degree45.png", h_default, sigma_arm_default, ARM_ANGLES3, PITCH_ANGLES_7)
#calc_modelled_emissivity()
#plot_interpolated_galactic_densities()