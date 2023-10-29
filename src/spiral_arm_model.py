import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import LinearNDInterpolator
from invdisttree import Invdisttree



# constants
h_default = 2.4                 # kpc, scale length of the disk. The value Higdon and Lingenfelter used
r_s = 7.6               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
rho_min = 2.9           # kpc, minimum distance from galactic center to the beginning of the spiral arms.
rho_max = 35            # kpc, maximum distance from galactic center to the end of the spiral arms.
sigma = 0.15            # kpc, scale height of the disk
sigma_arm_default = 0.5         # kpc, dispersion of the spiral arms
total_galactic_n_luminosity = 1.85e40       #total galactic N 2 luminosity in erg/s
measured_nii = 1.175e-4 # erg/s/cm^2/sr, measured N II 205 micron line intensity. Estimated from the graph in the paper
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
# note that the authors where this formula came from used different numbers, and got a value of 47 kpc^2.
# With these numbers, we get a value of 22.489 kpc^2
a_d = 2*np.pi*h_default**2 * ((1+rho_min/h_default)*np.exp(-rho_min/h_default) - (1+rho_max/h_default)*np.exp(-rho_max/h_default)) 
# starting angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
arm_angles = np.radians([70, 160, 250, 340])
# pitch angles for the spiral arms, respectively Norma-Cygnus(NC), Perseus(P), Sagittarius-Carina(SA), Scutum-Crux(SC)
pitch_angles = np.radians(np.array([13.5, 13.5, 13.5, 15.5]))
# the fractional contribution described in the text.
fractions = [0.18, 0.36, 0.18, 0.28]
number_of_end_points = 181 # number of points to use for the circular projection at the end points of the spiral arms
spiral_arm_names = ['Norma-Cygnus', 'Perseus', 'Sagittarius-Carina', 'Scutum-Crux']
fractional_contribution_default = [0.18, 0.32, 0.18, 0.32]


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


def generate_non_uniform_spacing(sigma_arm=sigma_arm_default, d_min=0.01, d_max=5, scaling=0.03):
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


def generate_gum_cygnus():
    # Cygnus parameters
    cygnus_distance = 1.45 # kpc
    cygnus_long = np.radians(80)
    cygnus_lat = np.radians(0)
    cygnus_radius = 0.075 # kpc
    # Gum parameters
    gum_distance = 0.33 # kpc
    gum_long = np.radians(262)
    gum_lat = np.radians(0)
    gum_radius = 0.03 # kpc
    
    fwhm = np.radians(7)
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    

def interpolate_density(grid_x, grid_y, method='linear', h=h_default, sigma_arm=sigma_arm_default):
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
    interpolated_densities = []
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
        interpolated_arm = griddata((x, y), density_spiral_arm, (grid_x, grid_y), method=method, fill_value=0)
        interpolated_densities.append(interpolated_arm)
    return np.array(interpolated_densities)


def plot_interpolated_galactic_densities(method='linear', h=h_default, sigma_arm=sigma_arm_default):
    """
    Returns:
        a plot of the interpolated galactic densities. The galactic densities are interpolated from the individual spiral arm densities
        the plot is a heatmap to indicate the density, and all spiral arms are plotted in the same plot
    """
    grid_x, grid_y = np.mgrid[-20:20:1000j, -20:20:1000j]
    total_galactic_densities = interpolate_density(grid_x, grid_y, method, h, sigma_arm)
    total_galactic_density = np.sum(total_galactic_densities, axis=0)
    #plot heatmap of the interpolated densities:
    plt.scatter(grid_x, grid_y, c=total_galactic_density.flatten(), cmap='viridis', s=20)
    plt.scatter(0, 0, c = 'magenta', s=2, label='Galactic centre')
    plt.scatter(0, r_s, c = 'gold', s=2, label='Sun')
    plt.gca().set_aspect('equal')
    plt.xlabel('Distance in kpc from the Galactic center')
    plt.ylabel('Distance in kpc from the Galactic center')
    plt.title('Heatmap of the interpolated densities of spiral arms in our model')
    plt.legend(loc='upper right')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    plt.savefig("output/interpolated_spiral_arms_density_model.png", dpi=600)


def calc_effective_area_per_spiral_arm(method='linear', h=h_default, sigma_arm=sigma_arm_default):
    """
    Calculates the effective area for each spiral arm. The density of each spiral arm is integrated over the entire galactic plane.
    The returned effective areas are in units of kpc^2, and appears in the same order as the spiral arms in arm_angles.
    """
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(sigma_arm) #d_min: minimum distance from the spiral arm
    d_x = 70 / 1000 # distance between each interpolated point in the x direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    d_y = 70 / 1000 # distance between each interpolated point in the y direction. 70 kpc is the diameter of the Milky Way, 1000 is the number of points
    grid_x, grid_y = np.mgrid[-35:35:1000j, -35:35:1000j]
    effective_area = []
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i])
        # generate the spiral arm points in spherical coordinates
        rho_coords, theta_coords = generate_spiral_arm_points_spherical_coords(rho, theta, pitch_angles[i], transverse_distances)
        # generate the spiral arm densities
        density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial, h)
        # in accordance with equation (9) in short.paper.2.0, the densities shall be scaled by their radial distance from the galactic centre
        density_spiral_arm *= rho_coords
        # Convert to Cartesian coordinates
        x = rho_coords*np.cos(theta_coords)
        y = rho_coords*np.sin(theta_coords)
        # calculate interpolated density for the spiral arm
        interpolated_density = griddata((x, y), density_spiral_arm, (grid_x, grid_y), method, fill_value=0)
        # add the interpolated density to the total galactic density
        effective_area = np.append(effective_area, np.sum(interpolated_density) * d_x * d_y)
    filepath = "output/effective_area_per_spiral_arm.txt"
    np.savetxt(filepath, effective_area)
    return effective_area


def calc_modelled_emissivity(fractional_contribution=fractional_contribution_default, method='linear', readfile = "true", h=h_default, sigma_arm=sigma_arm_default):
    print("Calculating modelled emissivity")
    if readfile == "true":
        effective_area = np.loadtxt("output/effective_area_per_spiral_arm.txt")
    else:
        effective_area = calc_effective_area_per_spiral_arm(method)
    # generate the set of coordinates
    dr = 0.01   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.1   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360 - dl, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max + 5, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # coordinates made. Now we need to calculate the density for each point
    height_distribution_values = height_distribution(z_grid)
    latitudinal_cosinus = np.cos(coordinates[:, 1])
    densities_as_func_of_long = np.zeros((len(pitch_angles), len(longitudes)))
    interpolated_densities = interpolate_density(x_grid, y_grid, method, h, sigma_arm)
    common_multiplication_factor = height_distribution_values * db * dl * latitudinal_cosinus/ (4 * np.pi * np.radians(1) * 10e4)
    ##############################################################################################################
    total_galactic_density = np.sum(interpolated_densities, axis=0)
    #plot heatmap of the interpolated densities:
    print("plotting...")
    """ plt.scatter(x_grid, y_grid, c=range(len(x_grid)), cmap='viridis', s=1)
    plt.show() """
    plt.scatter(x_grid, y_grid, c=total_galactic_density.flatten(), cmap='viridis', s=20)
    #plt.scatter(0, 0, c = 'magenta', s=2, label='Galactic centre')
    #plt.scatter(0, r_s, c = 'gold', s=2, label='Sun')
    plt.gca().set_aspect('equal')
    plt.xlabel('Distance in kpc from the Galactic center')
    plt.ylabel('Distance in kpc from the Galactic center')
    plt.title('Heatmap of the interpolated densities of spiral arms in our model')
    plt.legend(loc='upper right')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    plt.savefig("output/fakdifak2.png", dpi=600)
    
    ##############################################################################################################
    for i in range(len(arm_angles)):
        print("Calculating spiral arm number: ", i+1)
        interpolated_density_arm = interpolated_densities[i] * fractional_contribution[i] * effective_area[i] * common_multiplication_factor
        # reshape this 1D array into 2D array to facilitate for the summation over the different longitudes
        interpolated_density_arm = interpolated_density_arm.reshape((len(longitudes), len(radial_distances) * len(latitudes)))
        # sum up to get the density as a function of longitude
        density_distribution = interpolated_density_arm.sum(axis=1) # sum up all the values for the different longitudes
        densities_as_func_of_long[i] += density_distribution
    return longitudes, densities_as_func_of_long #* np.radians(5)) # devide by delta-b and delta-l in radians, respectively, for the averaging the paper mentions


def plot_modelled_emissivity_per_arm(fractional_contribution, method='linear', readfile = "true", filename = "output/modelled_emissivity.png", h=h_default, sigma_arm=sigma_arm_default):
    """
    Plots the modelled emissivity of the Galactic disk as a function of Galactic longitude.
    """
    longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, method, readfile)
    print("interpolated densities: ", densities_as_func_of_long.shape)
    plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[0]}")
    plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[1]}")
    plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[2]}")
    plt.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[3]}")
    plt.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
    print(np.sum(densities_as_func_of_long))
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    # Add parameter values as text labels
    plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.legend()
    plt.savefig(filename, dpi=1200)
    plt.show()


def plot_modelled_emissivity_total(fractional_contribution, method='linear', readfile = "true", filename = "output/modelled_emissivity.png", h=h_default, sigma_arm=sigma_arm_default):
    """
    Plots the modelled emissivity of the Galactic disk as a function of Galactic longitude.
    """
    longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution, method, readfile)
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
    fractional_contribution = [0.18, 0.32, 0.18, 0.32] # fractional contribution of each spiral arm to the total NII 205 micron line intensity
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
                               [0.18, 0.32, 0.18, 0.32]]
    # Create subplots for each arm
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    # Adjust the horizontal and vertical spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.12)
    # wspace=0.6 became too tight
    for i in range(num_rows):
        for j in range(num_cols):
            print("Calculating spiral arm number: ", i * num_cols + j + 1)
            ax = axes[i, j]
            longitudes, densities_as_func_of_long = calc_modelled_emissivity(fractional_contribution[i * num_cols + j], method, readfile)
            ax.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[0], label=f"NC. f={fractional_contribution[i * num_cols + j][0]}")
            ax.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[1], label=f"P. $\ $ f={fractional_contribution[i * num_cols + j][1]}")
            ax.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[2], label=f"SA. f={fractional_contribution[i * num_cols + j][2]}")
            ax.plot(np.linspace(0, 100, len(longitudes)), densities_as_func_of_long[3], label=f"SC. f={fractional_contribution[i * num_cols + j][3]}")
            ax.plot(np.linspace(0, 100, len(longitudes)), np.sum(densities_as_func_of_long, axis=0), label="Total")
            ax.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_A$ = {sigma_arm} kpc', transform=ax.transAxes, fontsize=8, color='black')
            # Redefine the x-axis labels to match the values in longitudes
            x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
            ax.set_xticks(np.linspace(0, 100, 13), x_ticks)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.set_xlabel("Galactic longitude l (degrees)")
            ax.set_ylabel("Modelled emissivity")
            ax.set_title("Modelled emissivity of the Galactic disk")
            ax.legend()         
    print("Done with plotting. Saving figure...") 
    plt.suptitle('Testing different values for the fractional contribution of each spiral arm')
    plt.savefig("output/test_fractional_contribution", dpi=1200, bbox_inches='tight')
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

def test_rho_theta_func():
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    print(longitudes)
    radiis = (0, 0.1, 0.5, 1)
    rhos_0 = rho_func(longitudes, 0, 0)
    thetas_0 = theta_func(longitudes, 0, 0)
    x_0 = rhos_0 * np.cos(thetas_0)
    y_0 = rhos_0 * np.sin(thetas_0)
    
    rhos_0_1 = rho_func(longitudes, 0, 0.1)
    thetas_0_1 = theta_func(longitudes, 0, 0.1)
    x_0_1 = rhos_0_1 * np.cos(thetas_0_1)
    y_0_1 = rhos_0_1 * np.sin(thetas_0_1)
    
    rhos_0_5 = rho_func(longitudes, 0, 0.5)
    thetas_0_5 = theta_func(longitudes, 0, 0.5)
    x_0_5 = rhos_0_5 * np.cos(thetas_0_5)
    y_0_5 = rhos_0_5 * np.sin(thetas_0_5)
    
    rhos_1 = rho_func(longitudes, 0, 1)
    thetas_1 = theta_func(longitudes, 0, 1)
    x_1 = rhos_1 * np.cos(thetas_1)
    y_1 = rhos_1 * np.sin(thetas_1)
    
    rhos_2 = rho_func(longitudes, 0, 2)
    thetas_2 = theta_func(longitudes, 0, 2)
    x_2 = rhos_2 * np.cos(thetas_2)
    y_2 = rhos_2 * np.sin(thetas_2)
    
    plt.scatter(x_0, y_0, cmap='viridis', c=range(len(y_0)), label="0")
    plt.scatter(x_0_1, y_0_1, cmap='viridis', c=range(len(y_0_1)), label="0.1")
    plt.scatter(x_0_5, y_0_5, cmap='viridis', c=range(len(y_0_5)), label="0.5")
    plt.scatter(x_1, y_1, cmap='viridis', c=range(len(y_1)), label="1")
    plt.scatter(x_2, y_2, cmap='viridis', c=range(len(y_2)), label="2")
    plt.gca().set_aspect('equal')
    plt.show()
    plt.legend()
    plt.show()


def test_interpolation_griddata_cubic(h=h_default, sigma_arm=sigma_arm_default):
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max + 5, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # make sample data
    d_theta = 0.01
    d_rho = 0.01
    rho = np.arange(0, 20, d_rho)
    theta = np.arange(0, 2*np.pi, d_theta)
    rho, theta = np.meshgrid(rho, theta, indexing='ij')
    coords = np.column_stack((rho.ravel(), theta.ravel()))  
    rho = coords[:, 0]
    theta = coords[:, 1]
    values = np.exp(-rho/h)
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    # interpolate data
    interpolated_densities = griddata((x, y), values, (x_grid, y_grid), method='cubic', fill_value=0)
    ###
    #interpolated_densities = griddata((rho, theta), values, (rho_coords_galaxy, theta_coords_galaxy), method='linear', fill_value=0)
    ###
    """ plt.scatter(x, y, c=values, cmap='viridis', s=20, label="original data")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig('output/test_interpolation_griddata12_original', dpi=1200) """
    plt.scatter(x_grid, y_grid, c=interpolated_densities.flatten(), cmap='viridis', s=1, label="interpolated data")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig('output/test_interpolation_griddata_cubic', dpi=1200)


def test_interpolation_griddata_spherical_linear(h=h_default, sigma_arm=sigma_arm_default):
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max + 5, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # make sample data
    d_theta = 0.01
    d_rho = 0.01
    rho = np.arange(0, 20, d_rho)
    theta = np.arange(0, 2*np.pi, d_theta)
    rho, theta = np.meshgrid(rho, theta, indexing='ij')
    coords = np.column_stack((rho.ravel(), theta.ravel()))  
    rho = coords[:, 0]
    theta = coords[:, 1]
    values = np.exp(-rho/h)
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    # interpolate data
    ###
    interpolated_densities = griddata((rho, theta), values, (rho_coords_galaxy, theta_coords_galaxy), method='linear', fill_value=0)
    ###
    plt.scatter(x_grid, y_grid, c=interpolated_densities.flatten(), cmap='viridis', s=1, label="interpolated data")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig('output/test_interpolation_spherical_linear', dpi=1200)
    

def test_interpolation_griddata_spherical_cubic(h=h_default, sigma_arm=sigma_arm_default):
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max + 5, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # make sample data
    d_theta = 0.01
    d_rho = 0.01
    rho = np.arange(0, 20, d_rho)
    theta = np.arange(0, 2*np.pi, d_theta)
    rho, theta = np.meshgrid(rho, theta, indexing='ij')
    coords = np.column_stack((rho.ravel(), theta.ravel()))  
    rho = coords[:, 0]
    theta = coords[:, 1]
    values = np.exp(-rho/h)
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    # interpolate data
    ###
    interpolated_densities = griddata((rho, theta), values, (rho_coords_galaxy, theta_coords_galaxy), method='cubic', fill_value=0)
    ###
    plt.scatter(x_grid, y_grid, c=interpolated_densities.flatten(), cmap='viridis', s=1, label="interpolated data")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig('output/test_interpolation_spherical_cubic', dpi=1200)


def test_interpolation_CloughTocher2DInterpolator(h=h_default, sigma_arm=sigma_arm_default):
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max + 5, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # make sample data
    d_theta = 0.01
    d_rho = 0.01
    rho = np.arange(0, 20, d_rho)
    theta = np.arange(0, 2*np.pi, d_theta)
    rho, theta = np.meshgrid(rho, theta, indexing='ij')
    coords = np.column_stack((rho.ravel(), theta.ravel()))  
    rho = coords[:, 0]
    theta = coords[:, 1]
    values = np.exp(-rho/h)
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    # interpolate data
    #interpolated_densities = griddata((x, y), values, (x_grid, y_grid), fill_value=0)
    interp = CloughTocher2DInterpolator(list(zip(x, y)), values)
    interpolated_densities = interp(x_grid, y_grid)
    plt.scatter(x_grid, y_grid, c=interpolated_densities, cmap='viridis', s=1, label="interpolated data")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig('output/test_interpolation_CloughTocher2DInterpolator', dpi=1200)


def test_interpolation_LinearNDInterpolator(h=h_default, sigma_arm=sigma_arm_default):
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max + 5, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # make sample data
    d_theta = 0.01
    d_rho = 0.01
    rho = np.arange(0, 20, d_rho)
    theta = np.arange(0, 2*np.pi, d_theta)
    rho, theta = np.meshgrid(rho, theta, indexing='ij')
    coords = np.column_stack((rho.ravel(), theta.ravel()))  
    rho = coords[:, 0]
    theta = coords[:, 1]
    values = np.exp(-rho/h)
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    # interpolate data
    #interpolated_densities = griddata((x, y), values, (x_grid, y_grid), fill_value=0)
    interp = LinearNDInterpolator(list(zip(x, y)), values)
    interpolated_densities = interp(x_grid, y_grid)
    plt.scatter(x_grid, y_grid, c=interpolated_densities, cmap='viridis', s=1, label="interpolated data")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig('output/test_interpolation_LinearNDInterpolator', dpi=1200)


def test_long_lat_rad_coords_generation():
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 0.5   # increments in dl (degrees):
    db = 0.1   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # Now I want to plot the x_grid and y_grid to see if they are correct
    # Use a colormap for the points. The plotted points will be colored according to the value of the array index
    # If the plotted points are correct, then the gradient shall go from theta=180 to theta=360 in the direction of the clockwise rotation
    # And alway go from the centre and outwards
    plt.scatter(x_grid, y_grid, c=range(len(x_grid)), cmap='viridis', s=1)
    plt.gca().set_aspect('equal')
    plt.savefig('output/test_long_lat_rad_coords_generation_r_s_2.png', dpi=1200)
    plt.show()


def test_invdisttree(fractional_contribution=fractional_contribution_default, method='linear', readfile = "true", h=h_default, sigma_arm=sigma_arm_default):
    
    N = 10000
    Ndim = 2
    Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 1  # weights ~ 1 / distance**p
    
    # generate the set of coordinates
    dr = 0.1   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitude_range = np.radians(3.5)
    latitudes = np.arange(-latitude_range, latitude_range, db)
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max + 5, dr) #r_s + rho_max + 5 is the maximum distance from the Sun to the outer edge of the Galaxy. +5 is due to the circular projection at the end points of the spiral arms
    # Create a meshgrid of all combinations
    lon_grid, lat_grid, radial_grid = np.meshgrid(longitudes, latitudes, radial_distances, indexing='ij')
    # Combine the grids into a 2-D array
    coordinates = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), radial_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    rho_coords_galaxy = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    theta_coords_galaxy = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    x_grid = rho_coords_galaxy * np.cos(theta_coords_galaxy)
    y_grid = rho_coords_galaxy * np.sin(theta_coords_galaxy)
    z_grid = coordinates[:, 2] * np.sin(coordinates[:, 1])
    # coordinates made. Now we need to calculate the density for each point
    height_distribution_values = height_distribution(z_grid)
    latitudinal_cosinus = np.cos(coordinates[:, 1])
    densities_as_func_of_long = np.zeros((len(pitch_angles), len(longitudes)))
    
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(sigma_arm) #d_min: minimum distance from the spiral arm
    interpolated_densities = []
    for i in range(len(arm_angles)):
        print("arm number: ", i)
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
        #interpolated_densities.append(griddata((x, y), density_spiral_arm, (grid_x, grid_y), method=method, fill_value=0))
        invdisttree = Invdisttree( np.array([x,y]).T, density_spiral_arm, leafsize=leafsize, stat=1 )
        interpol = invdisttree( np.array([x_grid, y_grid]).T, nnear=Nnear, eps=eps, p=p )
        interpolated_densities.append(interpol)
    #interpolated_densities = interpolate_density(x_grid, y_grid, method, h, sigma_arm)
    common_multiplication_factor = height_distribution_values * db * dl * latitudinal_cosinus/ (4 * np.pi * np.radians(1) * 10e4)
    ##############################################################################################################
    total_galactic_density = np.sum(interpolated_densities, axis=0)
    #plot heatmap of the interpolated densities:
    print("plotting...")
    plt.scatter(x_grid, y_grid, c=range(len(x_grid)), cmap='viridis', s=1)
    plt.show()
    plt.scatter(x_grid, y_grid, c=total_galactic_density.flatten(), cmap='viridis', s=20)
    #plt.scatter(0, 0, c = 'magenta', s=2, label='Galactic centre')
    #plt.scatter(0, r_s, c = 'gold', s=2, label='Sun')
    plt.gca().set_aspect('equal')
    plt.xlabel('Distance in kpc from the Galactic center')
    plt.ylabel('Distance in kpc from the Galactic center')
    plt.title('Heatmap of the interpolated densities of spiral arms in our model')
    plt.legend(loc='upper right')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    plt.savefig("output/fakdifak_invdisttree.png", dpi=600)
    

def test_spherical_interpolation(fractional_contribution=fractional_contribution_default, readfile="true", method='linear', h=h_default, sigma_arm=sigma_arm_default):
    #### 14
    print("Calculating modelled emissivity")
    if readfile == "true":
        effective_area = np.loadtxt("output/effective_area_per_spiral_arm.txt")
    else:
        effective_area = calc_effective_area_per_spiral_arm(method)
    ####
    drho = 0.01
    dtheta = 0.1
    db = 0.2 # and decrease this
    dl = 0.2 # should increase this
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360 -dl, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    print("longitudes shape: ", longitudes.shape)

    latitudes = np.radians(np.arange(-3.5, 3.5, db)) #18
    
    print("latidues shape: ", latitudes.shape)

    rhos = np.arange(0, 35, drho)
    print("rhos shape", rhos.shape)
    #thetas = np.radians(np.arange(0, 360 + dtheta, dtheta))
    ################ 15
    rhos, thetas, lats = np.meshgrid(rhos, longitudes, latitudes, indexing='ij')
    print("Meshed done")
    coordinates = np.column_stack((rhos.ravel(), thetas.ravel(), lats.ravel()))
    print("Column stacked done")
    print("Column stacked shape: ", coordinates.shape)

    print("calculating rho_gal")
    rho_gal = rho_func(coordinates[:, 1], coordinates[:, 2], coordinates[:, 0])
    print("calculating theta_gal")
    theta_gal = theta_func(coordinates[:, 1], coordinates[:, 2], coordinates[:, 0]) 
    ###################
    """ 
    thetas, lats, rhos = np.meshgrid(longitudes, latitudes, rhos, indexing='ij')
    print("Meshed done")
    coordinates = np.column_stack((thetas.ravel(), lats.ravel(), rhos.ravel()))
    print("Column stacked done")
    print("Column stacked shape: ", coordinates.shape)

    print("calculating rho_gal")
    rho_gal = rho_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    print("calculating theta_gal")
    theta_gal = theta_func(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]) 
    """
    #####################
    print("Calculating x_grid and y_grid")
    x_grid = rho_gal * np.cos(theta_gal)
    y_grid = rho_gal * np.sin(theta_gal)
    z_grid = coordinates[:, 0] * np.sin(coordinates[:, 2]) #9
    print(x_grid.shape, y_grid.shape, z_grid.shape)

    height_distribution_values = height_distribution(z_grid) #10
    latitudinal_cosinus = np.cos(coordinates[:, 1]) #11
    densities_as_func_of_long = np.zeros((len(pitch_angles), len(longitudes))) #12
    
    interpolated_density = interpolate_density(x_grid, y_grid)
    common_multiplication_factor = height_distribution_values * db * dl * latitudinal_cosinus/ (4 * np.pi * np.radians(1) * 10e4) #13
    # common_multiplication_factor = height_distribution_values * db * dl * latitudinal_cosinus/ (4 * np.pi * np.radians(1) * 10e4)
    total_galactic_density = np.sum(interpolated_density, axis=0)
    plt.scatter(x_grid, y_grid, c=total_galactic_density, cmap='viridis', s=1)
    plt.gca().set_aspect('equal')
    plt.savefig('output/test_spherical_interpolation_earth18.png', dpi=1200)
    """ ############################################################################################################## 16
    for i in range(len(arm_angles)):
        print("Calculating spiral arm number: ", i+1)
        interpolated_density_arm = interpolated_density[i] * fractional_contribution[i] * effective_area[i] * common_multiplication_factor
        # reshape this 1D array into 2D array to facilitate for the summation over the different longitudes
        interpolated_density_arm = interpolated_density_arm.reshape((len(longitudes), len(rhos) * len(latitudes)))
        # sum up to get the density as a function of longitude
        density_distribution = interpolated_density_arm.sum(axis=1) # sum up all the values for the different longitudes
        densities_as_func_of_long[i] += density_distribution
    return longitudes, densities_as_func_of_long #* np.radians(5)) # devide by delta-b and delta-l in radians, respectively, for the averaging the paper mentions """


#plot_interpolated_galactic_densities()
#plot_spiral_arms()
#calc_modelled_emissivity()
#plot_model_spiral_arm_densities()
#calc_effective_area_per_spiral_arm()
fractional_contribution = [0.18, 0.32, 0.18, 0.32] # fractional contribution of each spiral arm to the total NII 205 micron line intensity
#plot_modelled_emissivity_per_arm(fractional_contribution, 'linear', 'true', "output/modelled_emissivity_18_32_18_32.png")
#test_fractional_contribution()
#test_interpolation_method()
#calc_modelled_emissivity(fractional_contribution, 'cubic', 'true')
#plot_interpolated_galactic_densities()
#test_rho_theta_func()
#test_interpolation_griddata() 
#test_interpolation_method_interpolated_densities()
#print("long_lat_rad_coords_generation")
#test_long_lat_rad_coords_generation()
""" print("test_interpolation_griddata_cubic")
test_interpolation_griddata_cubic()
print("test_interpolation_griddata_spherical_linear")
test_interpolation_griddata_spherical_linear()
print("test_interpolation_griddata_spherical_cubic")
test_interpolation_griddata_spherical_cubic()
print("test_interpolation_CloughTocher2DInterpolator")
test_interpolation_CloughTocher2DInterpolator()
print("test_interpolation_LinearNDInterpolator")
test_interpolation_LinearNDInterpolator() """
#test_invdisttree()
test_spherical_interpolation()

# plot number 18: 220M points. Plot number 17: 25M points. 