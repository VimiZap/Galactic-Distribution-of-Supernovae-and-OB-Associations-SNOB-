import matplotlib.pyplot as plt
import numpy as np

inital_point = (3, 3)
points_to_rotate = np.array([[2, 1], [2.5, 2], [3.5, 4], [4, 5]])

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


def generate_non_uniform_spacing(d_min = 0.01, d_max = 5, scaling = 0.03):
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
    return d_rho


def generate_transverse_points(arm_medians, d_min = 0.01):
    """
    Args:
        arm_medians: array of distances \rho to the arm median
        d_min: minimum distance from the spiral arm
    Returns:
        a matrix of transverse points (rho's) for the spiral arm
    """
    d_rho = generate_non_uniform_spacing(d_min=d_min)
    cumsum = np.cumsum(d_rho)
    # for each point in arm_medians, generate the transverse points
    transverse = []
    for arm_median in arm_medians:
        transverse_points = arm_median - cumsum
        transverse_points = np.append(transverse_points, arm_median + cumsum)
        transverse.append(transverse_points)
    return np.array(transverse) 


############################################
def test():
    # These are then values for two points on a spiral arm
    rhos = np.array([1, 0.5])
    thetas = np.array([np.radians(30), np.radians(60)])
    rotation_point_1 = np.array([rhos[0]*np.cos(thetas[0]), rhos[0]*np.sin(thetas[0])])
    print(rotation_point_1)
    # transverse points:
    transverse = generate_transverse_points(rhos)
    print(transverse)
    # translate the transverse points into x and y coordinates, then rotate them and translate them back
    x1, y1 = transverse[0] * np.cos(thetas[0]), transverse[0] * np.sin(thetas[0])
    rotated_points = rotate_point(np.array([x1,y1]).T, np.radians(30), rotation_point_1)
    
    plt.scatter(x1, y1)
    plt.scatter(rotated_points[:, 0], rotated_points[:, 1])
    plt.scatter(rotation_point_1[0], rotation_point_1[1], c='black')
    plt.scatter(0, 0, c='black')
    plt.xlim(-4,6)
    plt.ylim(-5,6)
    plt.show()

def plot(points, angle):
    """
    Args:
        points: points to rotate
        angle: angle to rotate the points by
    Plots the points before and after rotation
    """
    rotated_points = rotate_point(points, angle, inital_point)
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(rotated_points[:, 0], rotated_points[:, 1])
    plt.show()

generate_non_uniform_spacing()
#plot(points_to_rotate, np.radians(30))
test()  

"""def generate_transverse_points_2(arm_medians, transverse_distances, thetas, pitch_angle):
   
    Args:
        arm_medians: array of distances \rho to the arm median
        transverse_distances: array of transverse distances from the arm median
        thetas: array of thetas for the arm median
        pitch_angle: pitch angle of the spiral arm
    Returns:
        a 3d array, where the first index is a given point on the spiral arm, the second index is the transverse point, and the third index is the x and y coordinates
    
    rotation_matrix = np.array([[np.cos(pitch_angle), -np.sin(pitch_angle)], [np.sin(pitch_angle), np.cos(pitch_angle)]])
    #x, y = transverse distances, 0
    y = np.zeros(len(transverse_distances))
    rotated_points = np.matmul(np.array([transverse_distances, y]).T, rotation_matrix.T) # basically one set of rotated points
    negative_points = -np.flipud(rotated_points)
    transverse_distances = np.append(negative_points, rotated_points, axis=0)
    print(transverse_distances.shape)
    print(transverse_distances)
    x_arms_medians = arm_medians * np.cos(thetas)
    y_arms_medians = arm_medians * np.sin(thetas)
    x_y_arms_medians = np.array([x_arms_medians, y_arms_medians]).T
    print(x_y_arms_medians.shape)
    arm_medians_reshaped = x_y_arms_medians[:, np.newaxis, :]
    #transverse_distances_reshaped = transverse_distances[:, :, np.newaxis]
    #(transverse_distances_reshaped.shape)
    print(arm_medians_reshaped.shape)
    total_shit = arm_medians_reshaped + transverse_distances
    print(total_shit.shape)
    return total_shit"""


"""def generate_transverse_points_2(arm_medians, transverse_distances, thetas, pitch_angle):
    
    Args:
        arm_medians: array of distances \rho to the arm median
        transverse_distances: array of transverse distances from the arm median
        thetas: array of thetas for the arm median
        pitch_angle: pitch angle of the spiral arm
    Returns:
        a 3d array, where the first index is a given point on the spiral arm, the second index is the transverse point, and the third index is the x and y coordinates
    
    ## Use broadcasting to perform the multiplication
    #result = array_n[:, np.newaxis] * array_m
    angle_cos = np.cos(thetas - pitch_angle)
    angle_sin = np.sin(thetas - pitch_angle)
    a = angle_cos[:, np.newaxis] * transverse_distances
    b = angle_sin[:, np.newaxis] * transverse_distances
    print("a: ", a.shape)
    print("angle cos: ", angle_cos.shape)
    rotated_points_2 = np.concatenate((a[:, :, np.newaxis], b[:, :, np.newaxis]), axis=2)
    print("rot_2: ", rotated_points_2.shape)
    print(rotated_points_2[0])
    negative_points_2 = -np.flip(rotated_points_2, axis=1)
    print(negative_points_2[0])
    transverse_distances_2 = np.append(negative_points_2, rotated_points_2, axis=1)
    print("transverse_distances_2: ", transverse_distances_2.shape)
    print(transverse_distances_2[0])
    #########################
    rotation_matrix = np.array([[np.cos(pitch_angle), -np.sin(pitch_angle)], [np.sin(pitch_angle), np.cos(pitch_angle)]])
    #x, y = transverse distances, 0
    y = np.zeros(len(transverse_distances))
    rotated_points = np.matmul(np.array([transverse_distances, y]).T, rotation_matrix.T) # basically one set of rotated points
    negative_points = -np.flipud(rotated_points)
    transverse_distances = np.append(negative_points, rotated_points, axis=0)
    print(transverse_distances.shape)
    print(transverse_distances_2.shape)
    #print(transverse_distances)
    x_arms_medians = arm_medians * np.cos(thetas)
    y_arms_medians = arm_medians * np.sin(thetas)
    x_y_arms_medians = np.array([x_arms_medians, y_arms_medians]).T
    print("x_y_arms_medians: ", x_y_arms_medians.shape)
    result_2 = transverse_distances_2 + x_y_arms_medians[:, np.newaxis, :]
    print("result_2: ", result_2.shape)
    arm_medians_reshaped = x_y_arms_medians[:, np.newaxis, :]
    #transverse_distances_reshaped = transverse_distances[:, :, np.newaxis]
    #(transverse_distances_reshaped.shape)
    print(arm_medians_reshaped.shape)
    total_shit = arm_medians_reshaped + transverse_distances
    print(total_shit.shape)

    return result_2"""