import numpy as np
import seaborn as sns
import logging
logging.basicConfig(level=logging.INFO)
import src.utilities.constants as const
import src.utilities.utilities as ut
import matplotlib.pyplot as plt
import src.spiral_arm_model as sam


def add_spiral_arms_to_ax(ax):
    """ Add the spiral arms to the plot
    
    Args:
        ax: axis. The axis to add the spiral arms to
    
    Returns:
        None
    """
    colors = sns.color_palette('bright', 7)
    for i in range(len(const.arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(const.arm_angles[i], const.pitch_angles[i], arm_index=i)
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        ax.plot(x, y, marker='o', markersize=1, color=colors[i], label=f'arm {i+1}') # plot the spiral arm medians
    return


def plot_spiral_arm_medians():
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    add_spiral_arms_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', s=15, zorder=11)
    ax.text(-1, 0.5, 'Galactic centre', fontsize=8, zorder=7)
    ax.set_aspect('equal')
    plt.legend()
    plt.show()
    plt.close()


def generate_spiral_arm_medians():
    spiral_arm_vectors = []
    spiral_arm_medians = []
    for i in range(len(const.arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(const.arm_angles[i], const.pitch_angles[i], arm_index=i)
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        delta_x = np.diff(x)
        delta_y = np.diff(y)
        tuple_xy = (x, y)
        spiral_arm_medians.append((x, y))
        spiral_arm_vectors.append((delta_x, delta_y))
    return spiral_arm_medians, spiral_arm_vectors


def test_plot_vector():
    spiral_arm_medians, spiral_arm_vectors = generate_spiral_arm_medians()
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    add_spiral_arms_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', s=15, zorder=11)
    ax.text(-1, 0.5, 'Galactic centre', fontsize=8, zorder=7)
    for i in range(len(spiral_arm_medians)):
        x, y = spiral_arm_medians[i]
        delta_x, delta_y = spiral_arm_vectors[i]
        for j in range(len(delta_x)):
            ax.arrow(x[j], y[j], delta_x[j], delta_y[j], head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.set_aspect('equal')
    plt.legend()
    plt.show()
    plt.close()


def calculate_vector(x, y, spiral_arm_medians, spiral_arm_vectors): # x, y are th coordinates of a random point in the plane
    min_distance = 1000
    min_spiral_arm_index = 0
    for i in range(len(spiral_arm_medians)):
        x_median, y_median = spiral_arm_medians[i]
        delta_x, delta_y = spiral_arm_vectors[i]
        for j in range(len(delta_x)):
            x1 = x_median[j]
            y1 = y_median[j]
            distance = np.sqrt((x - x1)**2 + (y - y1)**2)
            if distance < min_distance:
                min_distance = distance
                min_spiral_arm_index = i
                vector = (delta_x[j], delta_y[j])
    return vector, min_spiral_arm_index, min_distance


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def calculate_averaged_vector(x, y, spiral_arm_medians, spiral_arm_vectors):
    spiral_arm_median = spiral_arm_medians.copy()
    spiral_arm_vector = spiral_arm_vectors.copy()
    print(f'Point: ({x}, {y})')
    vector_1, index_1, min_distance_1 = calculate_vector(x, y, spiral_arm_median, spiral_arm_vector)
    print('index_1:', index_1)
    spiral_arm_median.pop(index_1), spiral_arm_vector.pop(index_1)
    vector_2, index_2, min_distance_2 = calculate_vector(x, y, spiral_arm_median, spiral_arm_vector)
    vector = (vector_1[0]/min_distance_1 + vector_2[0]/min_distance_2)/2, (vector_1[1]/min_distance_1 + vector_2[1]/min_distance_2)/2
    return normalize(vector) / 10


def test_calculate_vector():
    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    add_spiral_arms_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', s=15, zorder=11)
    ax.text(-1, 0.5, 'Galactic centre', fontsize=8, zorder=7)
    ax.set_aspect('equal')
    
    random_points = np.random.rand(16, 2)*12
    spiral_arm_medians, spiral_arm_vectors = generate_spiral_arm_medians()
    for i in range(random_points.shape[0]):
        x, y = random_points[i]
        vector = calculate_averaged_vector(x, y, spiral_arm_medians, spiral_arm_vectors)
        ax.arrow(x, y, vector[0], vector[1], head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.legend()
    plt.show()
    plt.close()


def gaussian_plane(mu_0, x, diffusion_const, time):
    return np.exp(-(x-mu_0)**2/(4*diffusion_const*time))


def gaussian_vertical(diffusion_const, time, H):
    return np.exp(-2*diffusion_const*time/H)


""" def cr_bubble():
    d_orthogonal = 2e26 / const.kpc**2 * const.year_in_seconds # kpc^2/year. Valid for E = 5 GeV 
    d_parallel = 5e28 / const.kpc**2 * const.year_in_seconds # kpc^2/year. Valid for E = 5 GeV 
    x = 0
    y = 0
    spiral_arm_medians, spiral_arm_vectors = generate_spiral_arm_medians()
    parallel_vector = calculate_averaged_vector(x, y, spiral_arm_medians, spiral_arm_vectors)
    orthogonal_vector = (-parallel_vector[1], parallel_vector[0])
    xs = np.linspace(-10, 10, 100)
    ys = np.linspace(-10, 10, 100)
    zs = np.linspace(-5, 5, 100)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    times = np.linspace(1, 100, 10)
    for t in times:
        print(f'time: {t}')
        densities = gaussian_plane(0, X, d_orthogonal, t) * gaussian_plane(0, Y, d_parallel, t) #* gaussian_vertical(d_orthogonal, t, 3)
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, const.r_s, 0, color='red', marker='o', label='Sun', s=10, zorder=11)
        ax.scatter(0, 0, 0, color='black', marker='o', s=15, zorder=11)
        ax.set_aspect('equal')
        ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=densities.ravel(), cmap='viridis', s=1)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-5, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.colorbar()
        print('show')
        plt.show()
        plt.close()
        break """

def cr_bubble():
    d_orthogonal = 2e26 / const.kpc**2 * const.year_in_seconds # kpc^2/year. Valid for E = 5 GeV 
    d_parallel = 5e28 / const.kpc**2 * const.year_in_seconds # kpc^2/year. Valid for E = 5 GeV 
    x = 0
    y = 0
    spiral_arm_medians, spiral_arm_vectors = generate_spiral_arm_medians()
    parallel_vector = calculate_averaged_vector(x, y, spiral_arm_medians, spiral_arm_vectors)
    orthogonal_vector = (-parallel_vector[1], parallel_vector[0])
    xs = np.linspace(-10, 10, 100)
    ys = np.linspace(-10, 10, 100)
    zs = np.linspace(-5, 5, 100)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    times = np.linspace(1, 100, 10)
    for t in times:
        print(f'time: {t}')
        densities = gaussian_plane(0, X, d_orthogonal, t) * gaussian_plane(0, Y, d_parallel, t) * gaussian_vertical(d_orthogonal, t, 3)
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, const.r_s, 0, color='red', marker='o', label='Sun', s=10, zorder=11)
        ax.scatter(0, 0, 0, color='black', marker='o', s=15, zorder=11)
        ax.set_aspect('equal')
        #ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=densities.ravel(), cmap='viridis', s=1)
        ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=1)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-5, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        print('show')
        plt.show()
        plt.close()
        break  

        

    """ plt.figure(figsize=(16, 6))
    ax = plt.gca()
    add_spiral_arms_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', s=15, zorder=11)
    ax.text(-1, 0.5, 'Galactic centre', fontsize=8, zorder=7)
    ax.set_aspect('equal')
    ax.arrow(x, y, parallel_vector[0], parallel_vector[1], head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(x, y, orthogonal_vector[0], orthogonal_vector[1], head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.show()
    plt.close() """



#test_plot_vector()
#test_calculate_vector()
cr_bubble()  

