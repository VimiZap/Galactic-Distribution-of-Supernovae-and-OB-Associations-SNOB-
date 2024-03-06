import numpy as np
import logging




def test_plot_density_distribution_piss():
    """ Function to test the density distribution of the Milky Way. Plots both the unweighted, analytical density distribution and the weighted, modelled emissivity from which the associations are drawn.
    
    Args:
        None
    Returns:
        Saves two plots in the output folder
    """
    # let's make a plot for the density distribution of the Milky Way, to see if these maps actually reproduce the expected desnity distribution
    logging.info("Beginning to load the data for the uniform spiral arm density distribution")
    num_lats = len(np.lib.format.open_memmap(f'{GALAXY_DATA}/latitudes.npy'))
    num_rads = len(np.lib.format.open_memmap(f'{GALAXY_DATA}/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap(f'{GALAXY_DATA}/longitudes.npy'))
    # Load the data for the uniform spiral arm density distribution:
    NUM_INTERPOLANT_FILES = 4
    uniform_spiral_arm_density_distribution = np.load(f'{GALAXY_DATA}/interpolated_arm_0.npy')
    for i in range(NUM_INTERPOLANT_FILES - 1):
        uniform_spiral_arm_density_distribution += np.lib.format.open_memmap(f'{GALAXY_DATA}/interpolated_arm_{i+1}.npy')
    uniform_spiral_arm_density_distribution = np.reshape(uniform_spiral_arm_density_distribution, (num_rads, num_longs, num_lats))
    uniform_spiral_arm_density_distribution = np.sum(uniform_spiral_arm_density_distribution, axis=2)
    uniform_spiral_arm_density_distribution = uniform_spiral_arm_density_distribution.ravel()
    uniform_spiral_arm_density_distribution /= np.max(uniform_spiral_arm_density_distribution) # normalize this so that the maximum value is 1
    
    # Load the grid-data:
    x_grid = np.load(f'{GALAXY_DATA}/x_grid.npy')
    x_grid = np.reshape(x_grid, (num_rads, num_longs, num_lats))
    x_grid = x_grid[:,:,0]
    x_grid = x_grid.ravel()
    y_grid = np.load(f'{GALAXY_DATA}/y_grid.npy')
    y_grid = np.reshape(y_grid, (num_rads, num_longs, num_lats))
    y_grid = y_grid[:,:,0]
    y_grid = y_grid.ravel()
    logging.info("Loaded the data. Beginning to plot the figure")
    # Plot the uniform spiral arm density distribution:
    plot_density_distribution(x_grid, y_grid, uniform_spiral_arm_density_distribution, 'uniform')
    logging.info("Saved density map. Now drawing associations and plotting them based on this map")
    # Draw associations and plot them based on the density map
    rng = np.random.default_rng()
    NUM_ASC = 10000
    uniform_spiral_arm_density_distribution /= np.sum(uniform_spiral_arm_density_distribution) # normalize to unity
    grid_index = rng.choice(a=len(uniform_spiral_arm_density_distribution), size=NUM_ASC, p=uniform_spiral_arm_density_distribution, replace=False) #replace = False means that the same index cannot be drawn twice
    x = x_grid[grid_index]
    y = y_grid[grid_index]
    logging.info("Associations drawn. Beginning to plot the figure")
    plot_drawn_associations(x, y, NUM_ASC, 'test_association_placement_uniform.png')
    logging.info("Done saving the figure")
    del uniform_spiral_arm_density_distribution
    gc.collect()

    # Plot the weigthed density distribution:
    logging.info("Beginning to load the data for the emissivity")
    emissivity = np.load(f'{GALAXY_DATA}/interpolated_arm_emissivity_0.npy')
    for i in range(NUM_INTERPOLANT_FILES - 1):
        emissivity += np.lib.format.open_memmap(f'{GALAXY_DATA}/interpolated_arm_emissivity_{i+1}.npy')
    emissivity = np.reshape(emissivity, (num_rads, num_longs, num_lats))
    emissivity = np.sum(emissivity, axis=2)
    emissivity = emissivity.ravel()
    emissivity /= np.max(emissivity) # normalize this so that the maximum value is 1
    logging.info("Loaded the data. Beginning to plot the figure")
    plot_density_distribution(x_grid, y_grid, emissivity, 'emissivity')
    logging.info("Saved density map. Now drawing associations and plotting them based on this map")
    emissivity /= np.sum(emissivity) # normalize to unity
    grid_index = rng.choice(a=len(emissivity), size=NUM_ASC, p=emissivity, replace=False) #replace = False means that the same index cannot be drawn twice
    x = x_grid[grid_index]
    y = y_grid[grid_index]
    logging.info("Associations drawn. Beginning to plot the figure")
    plot_drawn_associations(x, y, NUM_ASC, 'test_association_placement_emissivity.png')
    return