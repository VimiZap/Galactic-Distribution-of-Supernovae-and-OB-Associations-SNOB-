import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interpn
import seaborn as sns


# constants
h = 2.5                 # kpc, scale length of the disk. The value Higdon and Lingenfelter used
r_s = 7.6               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
rho_min = 2.9           # kpc, minimum distance from galactic center to the beginning of the spiral arms.
rho_max = 35            # kpc, maximum distance from galactic center to the end of the spiral arms.
sigma = 0.15            # kpc, scale height of the disk
total_galactic_n_luminosity = 1.85e40       #total galactic N 2 luminosity in erg/s
measured_nii = 1.175e-4 # erg/s/cm^2/sr, measured N II 205 micron line intensity. Estimated from the graph in the paper
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
# note that the authors where this formula came from used different numbers, and got a value of 47 kpc^2.
# With these numbers, we get a value of 22.489 kpc^2
a_d = 2*np.pi*h**2 * ((1+rho_min/h)*np.exp(-rho_min/h) - (1+rho_max/h)*np.exp(-rho_max/h)) 
# starting angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
arm_angles = np.radians([70, 160, 250, 340])
# pitch angles for the spiral arms, respectively Norma-Cygnus, Perseus, Sagittarius-Carina, Scutum-Crux
pitch_angles = np.radians(np.array([13.5, 13.5, 13.5, 15.5]))
number_of_end_points = 181 # number of points to use for the circular projection at the end points of the spiral arms


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


def arm_median_density(rho): 
    """
    Args:
        rho: distance from the Galactic center
        h: scale length of the disk
    Returns:
        density of the disk at a distance rho from the Galactic center
    """
    return np.exp(-rho/h)


def generate_non_uniform_spacing(sigma_arm = 0.5, d_min = 0.01, d_max = 5, scaling = 0.03):
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
    print(transverse_distances[-1])
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



def plot_spiral_arms():
    """
    Plots the spiral arms, both the medians and also the transverse points. 
    """
    plt.scatter(0,0, c='magenta')
    colours = ['navy', 'darkgreen', 'darkorange', 'purple'] # one colour per arm
    transverse_distances, transverse_densities = generate_non_uniform_spacing(d_min=0.01) #d_min: minimum distance from the spiral arm
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
    # add them together to get the spherical coordinates in a 3D array. Axis 0 is the point along the spiral arm median, axis 1 is the transverse point, axis 2 is the rho and theta coordinates
    spherical_transverse_points = np.concatenate((radial_transverse_points[:, :, np.newaxis], angular_transverse_points[:, :, np.newaxis]), axis=2)
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


def generate_spiral_arm_densities(rho, transverse_densities_initial):
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
    arm_median_densities = arm_median_density(rho) #1D array
    # calculate the transverse densities for the arm. Does not contain the contrbution from the circular projection at the end points, but has the arm median density
    arm_transverse_densities = transverse_densities * arm_median_densities[:, np.newaxis] #2D array
    # calculate the densities for the end points projected in a circle around the end points
    density_start_arm = transverse_densities_initial * arm_median_densities[0] # this is a 1D array, but the same values goes for every index in start_arm along axis 0
    density_end_arm = transverse_densities_initial * arm_median_densities[-1] # this is a 1D array, but the same values goes for every index in end_arm along axis 0
    density_spiral_arm = np.concatenate([np.tile(density_start_arm, number_of_end_points), arm_transverse_densities.flatten(), np.tile(density_end_arm, number_of_end_points)])
    return density_spiral_arm
    

def plot_model_spiral_arm_densities():
    """
    Returns:
        a plot of the modelled spiral arm densities. Each arm is plotted in a separate subplot as a heatmap to indicate the density
    """
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(d_min=0.01)
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
            density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial)
            # Convert to Cartesian coordinates and plot the scatter plot
            x = rho_coords * np.cos(theta_coords)
            y = rho_coords * np.sin(theta_coords)
            ax.set_xlim(-40, 40)
            ax.set_ylim(-40, 40)
            ax.set_aspect('equal', adjustable='box')
            ax.scatter(x, y, c=density_spiral_arm, s=20)
            ax.set_xlabel('Distance in kpc from the Galactic center')
            ax.set_ylabel('Distance in kpc from the Galactic center')
            ax.set_title(f'Spiral Arm {i * num_cols + j + 1}')
    # Add a colorbar
    cbar = fig.colorbar(ax.collections[0], ax=axes, orientation='vertical')
    cbar.set_label('Density')
    plt.suptitle('Heatmap of the densities of spiral arms in our model')
    plt.savefig("output/spiral_arms_density_model.png", dpi=300)
    #plt.show()  # To display the plot


def plot_interpolated_galactic_densities():
    """
    Returns:
        a plot of the interpolated galactic densities. The galactic densities are interpolated from the individual spiral arm densities
        the plot is a heatmap to indicate the density, and all spiral arms are plotted in the same plot
    """
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(d_min=0.01) #d_min: minimum distance from the spiral arm
    grid_x, grid_y = np.mgrid[-35:35:1000j, -35:35:1000j]
    total_galactic_density = np.zeros((1000, 1000))
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = spiral_arm_medians(arm_angles[i], pitch_angles[i])
        # generate the spiral arm points in spherical coordinates
        rho_coords, theta_coords = generate_spiral_arm_points_spherical_coords(rho, theta, pitch_angles[i], transverse_distances)
        # generate the spiral arm densities
        density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial)
        # Convert to Cartesian coordinates
        x = rho_coords*np.cos(theta_coords)
        y = rho_coords*np.sin(theta_coords)
        # calculate interpolated density for the spiral arm
        interpolated_density = griddata((x, y), density_spiral_arm, (grid_x, grid_y), method='linear', fill_value=0)
        # add the interpolated density to the total galactic density
        total_galactic_density += interpolated_density
    #plot heatmap of the interpolated densities:
    plt.scatter(grid_x, grid_y, c=total_galactic_density.flatten(), cmap='viridis', s=20)
    plt.gca().set_aspect('equal')
    plt.xlabel('Distance in kpc from the Galactic center')
    plt.ylabel('Distance in kpc from the Galactic center')
    plt.title('Heatmap of the interpolated densities of spiral arms in our model')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    plt.savefig("output/interpolated_spiral_arms_density_model.png", dpi=600)


def calc_effective_area_per_spiral_arm():
    """
    Calculates the effective area for each spiral arm. The density of each spiral arm is integrated over the entire galactic plane.
    The returned effective areas are in units of kpc^2, and appears in the same order as the spiral arms in arm_angles.
    """
    transverse_distances, transverse_densities_initial = generate_non_uniform_spacing(d_min=0.01) #d_min: minimum distance from the spiral arm
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
        density_spiral_arm = generate_spiral_arm_densities(rho, transverse_densities_initial)
        # Convert to Cartesian coordinates
        x = rho_coords*np.cos(theta_coords)
        y = rho_coords*np.sin(theta_coords)
        # calculate interpolated density for the spiral arm
        interpolated_density = griddata((x, y), density_spiral_arm, (grid_x, grid_y), method='linear', fill_value=0)
        # add the interpolated density to the total galactic density
        effective_area = np.append(effective_area, np.sum(interpolated_density) * d_x * d_y)
    return effective_area

            

plot_interpolated_galactic_densities()
#plot_spiral_arms()
#calc_modelled_emissivity()
#plot_model_spiral_arm_densities()
