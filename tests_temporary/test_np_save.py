import numpy as np
import matplotlib.pyplot as plt
import time
r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to run.")
        return result
    return wrapper

def count_negative_values(arr):
    negative_values = arr[arr < 0]
    return len(negative_values), np.average(negative_values)


@timing_decorator
def test_time_draw_postitions_from_entire_array():
    # test_time_draw_postitions_from_entire_array took 1083.423133 seconds to run.
    data = np.load('output\galaxy_data\interpolated_densities.npy')
    x = np.load('output/galaxy_data/x_grid.npy')
    data = data/np.sum(data)
    for _ in range(1000):
        x = np.random.choice(a=len(data), size=1, p=data)

@timing_decorator
def test_time_draw_positions_rad_long_lat():
    #test_time_draw_positions_rad_long_lat took 0.900275 seconds to run.
    # positions:
    x_grid = np.load('output/galaxy_data/x_grid.npy')
    y_grid = np.load('output/galaxy_data/y_grid.npy')
    z_grid = np.load('output/galaxy_data/z_grid.npy')
    # densities:
    densities_longitudinal = np.load('output\galaxy_data\densities_longitudinal.npy')
    densities_longitudinal = densities_longitudinal/np.sum(densities_longitudinal)
    densities_lat = np.load('output\galaxy_data\densities_lat.npy')
    densities_lat = densities_lat/np.sum(densities_lat, axis=1, keepdims=True)
    rad_densities = np.load('output\galaxy_data\densities_rad.npy')
    rad_densities = rad_densities/np.sum(rad_densities, axis=0, keepdims=True)
    """ print(rad_densities.shape)
    print(rad_densities[:,0,0].shape)
    print(densities_longitudinal.shape)
    print(densities_lat.shape)
    print(np.sum(densities_lat, axis=1, keepdims=True).shape)
    print(densities_lat[0].ravel()) """
    longs = []
    lats = []
    rads = []
    xs = []
    ys = []
    dl = 0.2   # increments in dl (degrees):
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    db = 0.5   # increments in db (degrees):
    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitudes = np.radians(np.arange(-5, 5 + db, db))
    for _ in range(10000):
        # lol I am drawing the DENSITIES here, not the actual galactic position!
        long_index = np.random.choice(a=len(densities_longitudinal), size=1, p=densities_longitudinal )
        lat_index = np.random.choice(a=len(densities_lat[long_index].ravel()), size=1, p=densities_lat[long_index].ravel() )
        radial_index = np.random.choice(a=len(rad_densities[:,long_index,lat_index].ravel()), size=1, p=rad_densities[:, long_index, lat_index].ravel() )
        grid_index = radial_index * len(longitudes) * len(latitudes) + long_index * len(latitudes) + lat_index
        target_x = x_grid[grid_index]
        target_y = y_grid[grid_index]
        target_z = z_grid[grid_index]
        xs.append(target_x)
        ys.append(target_y)
    plt.scatter(xs, ys, s=1, color='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Distance from the Galactic Centre (kpc)")
    plt.ylabel("Distance from the Galactic Centre (kpc)")
    plt.suptitle("Associations drawn from the NII density distribution of the Milky Way")
    plt.title(f"Made with {len(xs)} associations")
    plt.savefig("output/galaxy_tests/positions_from_density_distribution.png", dpi=1200)     # save plot in the output folder
    plt.show()
    
def test_data_files():
    x_grid = np.load('output/galaxy_data/x_grid.npy')
    y_grid = np.load('output/galaxy_data/y_grid.npy')
    z_grid = np.load('output/galaxy_data/z_grid.npy')
    plt.scatter(x_grid, y_grid, s=1)
    plt.show()


test_time_draw_positions_rad_long_lat()
#test_data_files()