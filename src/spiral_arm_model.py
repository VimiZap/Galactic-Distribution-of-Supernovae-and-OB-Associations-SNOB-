import numpy as np
import matplotlib.pyplot as plt

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


def transverse_arm_distribution(delta, sigma_arm):
    """
    Args:
        delta: transverse distance from the medians of the modelled spiral arm
        sigma_arm: dispersion of the spiral arm
    Returns:
        the fall off of spiral arm populations transverse an arm median
    """
    return np.exp(-0.5 * delta**2 / sigma_arm**2)  / (np.sqrt(2*np.pi) * sigma_arm) # in the paper, they do not include this normalization factor for some reason


def rotate_point(point, angle, rotation_point=(0, 0)):
    """
    Args:
        point: point or points to rotate. (x,y) coordinates, (n, 2) matrix with n = number of points
        angle: angle to rotate the point by. Radians
        rotation_point: the point to rotate around. (x,y) coordinates
    Returns:
        rotated point
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.matmul(point - rotation_point, rotation_matrix.T) + rotation_point


def generate_non_uniform_spacing(sigma_arm = 0.5, d_min = 0.01, d_max = 5, scaling = 0.03):
    """
    Args:
        d_min: minimum distance from the spiral arm
        d_max: maximum distance from the spiral arm
        scaling: scaling of the exponential distribution. A value of 0.03 generates between 150-200 points, for the most part 160-180 points
    Returns:
        an array of incremental distances for rho. The sum of all elements in the array is equal to 4.99 (d_max - d_min).
        To get a non-uniform spacing of points, an exponential distribution is used.
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
    transverse_densities = transverse_arm_distribution(transverse_distances, sigma_arm)
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
        a 
    """
    angles_arc = np.linspace(0, np.pi, num=180) + theta - pitch_angle 
    if point == 'start':
        angles_arc += np.pi
    x_arc = rho * np.cos(theta) + transverse_distances * np.cos(angles_arc)[:, np.newaxis]
    y_arc = rho * np.sin(theta) + transverse_distances * np.sin(angles_arc)[:, np.newaxis]
    #x = np.cos(thetas)[:, np.newaxis] * rhos
    #y = np.sin(thetas)[:, np.newaxis] * rhos
    return np.array([x_arc, y_arc]).T



def plot_spiral_arms():
    """
    Plots the spiral arms
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
        # Flatten the 3D array into 2D arrays
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

plot_spiral_arms()
