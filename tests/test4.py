""" import numpy as np
import matplotlib.pyplot as plt

def generate_uniform_sphere(grid_size, radius):
    # Create a 3D grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    x, y, z = np.meshgrid(x, y, z)

    # Mask points outside the sphere
    mask = x**2 + y**2 + z**2 <= radius**2
    print(mask.shape)

    # Keep points within the sphere
    x_inside = x[mask]
    y_inside = y[mask]
    z_inside = z[mask]
    print(x.shape, x_inside.shape)
    print(y.shape, y_inside.shape)
    print(z.shape, z_inside.shape)
    # Density for the sphere
    density = np.ones_like(x_inside)
    density = density.sum(axis = 2)

    

    return x_inside, y_inside, z_inside

def plot_sphere(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Uniform Sphere on a Regular Grid')
    plt.show()

# Define grid size and sphere radius
grid_size = 20
sphere_radius = 1.0

# Generate uniform sphere on a regular grid
x, y, z = generate_uniform_sphere(grid_size, sphere_radius)

# Plot the sphere
plot_sphere(x, y, z)
 """
r_s = 8.178 # kpc
 
import numpy as np
import matplotlib.pyplot as plt
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


def plot_density(x, y, z, density_values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=density_values, marker='o', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Density within the Sphere')
    plt.show()

""" # Define grid size and sphere radius
grid_size = 1
sphere_radius = 1.0

# Generate uniform sphere on a regular grid with density values
x, y, z, density_values = generate_uniform_sphere(sphere_radius)

# Sum up z-values for each column in the xy plane
summed_x, summed_y, summed_z = sum_z_values(x, y, density_values)

# Plot the original sphere with density values
plot_density(x, y, z, density_values)

# Plot the summed z-values in the xy plane
plt.figure()
plt.scatter(summed_x, summed_y, c=summed_z, cmap='viridis', marker='o', s=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Summed Z-Values in the XY Plane')
plt.colorbar(label='Summed Z-Values')
plt.show() """
def plot_gum_cyg():
    summed_c_x, summed_c_y, densities_c, summed_g_x, summed_g_y, densities_g = generate_gum_cygnus()
    plt.figure()
    plt.scatter(summed_c_x, summed_c_y, c=densities_c, cmap='viridis', marker='o', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Summed Z-Values in the XY Plane for Cygnus')
    plt.colorbar(label='Summed Z-Values')
    plt.show()
    plt.close()
    plt.figure()
    plt.scatter(summed_g_x, summed_g_y, c=densities_g, cmap='viridis', marker='o', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Summed Z-Values in the XY Plane for Gum')
    plt.colorbar(label='Summed Z-Values')
    plt.show()
plot_gum_cyg()