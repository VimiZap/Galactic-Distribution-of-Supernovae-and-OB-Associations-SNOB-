import numpy as np
import matplotlib.pyplot as plt

r_s = 8.178 # kpc

def gaussian_distribution(x, sigma):
    return np.exp(-0.5 * x**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)

# what I must have here is the sphere generated on a xyz grid, so that I can sum up the densities on the plane
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

    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 20)
    cygnus_radians = np.linspace(0, cygnus_radius, 10)
    gum_radians = np.linspace(0, gum_radius, 10)
    #radial_grid, lon_grid, lat_grid = np.meshgrid(radial_distances, longitudes, latitudes, indexing='ij')
    
    c_r_grid, c_theta_grid, c_phi_grid = np.meshgrid(cygnus_radians, theta, phi, indexing='ij')
    g_r_grid, g_theta_grid, g_phi_grid = np.meshgrid(gum_radians, theta, phi, indexing='ij')

    c_density = gaussian_distribution(c_r_grid, std)
    #c_coordinates = np.column_stack((c_r_grid.ravel(), c_phi_grid.ravel(), c_theta_grid.ravel()))     # Now 'coordinates' is a 2-D array with all combinations of (longitude, latitude, radial_distance)
    g_density = gaussian_distribution(g_r_grid, std)
    print(c_density.shape, c_r_grid.shape, c_theta_grid.shape, c_phi_grid.shape)
    cygnus_x = c_r_grid * np.sin(c_theta_grid) * np.cos(c_phi_grid) + cygnus_distance * np.cos(cygnus_lat) * np.sin(cygnus_long) 
    cygnus_y = c_r_grid * np.sin(c_theta_grid) * np.sin(c_phi_grid) + cygnus_distance * np.cos(cygnus_lat) * np.cos(cygnus_long) + r_s
    cygnus_z = c_r_grid * np.cos(c_theta_grid) + cygnus_distance * np.sin(cygnus_lat)
    
    gum_x = g_r_grid * np.sin(g_theta_grid) * np.cos(g_phi_grid) + gum_distance * np.cos(gum_lat) * np.sin(gum_long)
    gum_y = g_r_grid * np.sin(g_theta_grid) * np.sin(g_phi_grid) + gum_distance * np.cos(gum_lat) * np.cos(gum_long)
    gum_z = g_r_grid * np.cos(g_theta_grid) + gum_distance * np.sin(gum_lat) 

    print(cygnus_x.ravel().shape, c_density.ravel().shape) # bare tull
    print(c_density.shape)

    c_coords = np.column_stack((cygnus_x.ravel(), cygnus_y.ravel()))
    g_coords = np.column_stack((gum_x.ravel(), gum_y.ravel()))
    # Create a 3D scatter plot
    """ fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cygnus_x.ravel(), cygnus_y.ravel(), cygnus_z.ravel(), c=c_density.ravel(), marker='o', label='Sphere')

    # Set axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label') """
    plt.scatter(cygnus_x.ravel(), cygnus_y.ravel(), c=c_density.ravel(), label='Cygnus')
    plt.legend()
    plt.show()
    return c_coords, g_coords, c_density.ravel(), g_density.ravel()

c_coords, g_coords, c_density, g_density = generate_gum_cygnus()
print(c_coords.shape, g_coords.shape, c_density.shape, g_density.shape)
print(c_coords[:,0].shape, c_coords[:,1].shape, c_density.shape)