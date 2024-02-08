import numpy as np
import matplotlib.pyplot as plt
import gc
import supernovae_class as sn
import association_class as ass
import galaxy_class as galaxy
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
import os
import time
import logging

WORK_DIRECTORY = '/work/paradoxx/viktormi/output'

N = 10e4 # number of associations in the Galaxy
T = 20 # simulation run time in Myrs
star_formation_episodes = [1, 3, 5]
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes
r_s = 8.178

def generate_galaxy(n, c, T):
    # n = number of associations
    # c = number of star formation episodes
    association_array = []
    for i in range(n):
        association_array.append(ass.Association(c, T))
    return np.array(association_array)

@np.vectorize
def vectorized_number_sn(association):
    return association.number_sn

@np.vectorize
def vectorized_longitudes(association):
    return association.longitudes

@np.vectorize
def vectorized_exploded_sn(association):
    return association.exploded_sn

def plot_cum_snp_cluster_distr(galaxies, C):
    # P(<N*): the cumultative distribution of steller clusters as function of number of snp's
    # n = number of associations
    # c = array with the value of c 
    for i in range(len(C)):
        print('i = ', i)
        n = galaxies[i].num_asc
        association_array_num_sn = vectorized_number_sn(galaxies[i].galaxy)
        num_bins = int(np.ceil(max(association_array_num_sn))) # minimum number of stars = 1
        counts, _ = np.histogram(association_array_num_sn, bins=range(1, num_bins, 1))
        cumulative = (n - np.cumsum(counts))/n # cumulative distribution, normalized
        #plt.plot(range(1, num_bins-1, 1), cumulative, label="Number of star formation episodes = " + str(C[i]))
        plt.plot(range(1, num_bins-1, 1), cumulative, label= fr"{star_formation_episodes[i]} episodes. Avg. number of SN's: {np.average(association_array_num_sn):.2f}")
        

    plt.xscale("log")
    plt.xlim(1, num_bins + 3000) # set the x axis limits
    plt.ylim(0, 1) # set the y axis limits
    plt.xlabel("Number of SNPs")
    plt.ylabel("Cumulative distribution. P(N > x)")
    plt.suptitle("Monte Carlo simulation of temporal clustering of SNPs")
    plt.title(f"Made with {n} associations")
    plt.legend()
    plt.savefig("output/galaxy_tests/temporal_clustering.png", dpi=1200)     # save plot in the output folder
    plt.close()

def sum_pairwise(a):
    paired_data = a.reshape(-1, 2)
    # Sum along the specified axis (axis=1 sums up each row)
    result = np.sum(paired_data, axis=1)
    return result


def rearange_data(data):
    # rearange data to be plotted in desired format. Also does the summation
    middle = int(len(data)/2)
    data_centre_left = data[0]
    data_left = sum_pairwise(data[1:middle-1])
    data_left_edge = data[middle-1]
    data_right_edge = data[middle]
    data_edge = (data_right_edge + data_left_edge)
    data_right = sum_pairwise(data[middle+1:-1])
    data_centre_right = data[-1]
    data_centre = (data_centre_left + data_centre_right)
    rearanged_data = np.concatenate(([data_edge], data_left[::-1], [data_centre], data_right[::-1], [data_edge]))
    return rearanged_data


def plot_hist_data(hist, filename):
    # Create bin_edges
    bin_edges_central = np.arange(2.5, 360, 5)
    bin_edges = np.concatenate(([0], bin_edges_central, [360]))
    # Plot using stairs
    plt.stairs(values=hist, edges=bin_edges, fill=False, color='black')
    # Set up the plot
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3))
    plt.xlabel('Galactic longitude (degrees)')
    plt.xlim(0, 360)
    plt.ylabel('Line intensity in nW m$^{-2}$ sr$^{-1}$')
    plt.title("N+ line intensity vs Galactic longitude")
    # Save the plot
    plt.savefig(filename, dpi=1200)
    plt.close()


def plot_sn_as_func_of_long(galaxy):
    logging.info("Plotting the probability density function of SNPs as function of longitude")
    exploded_sn_long = np.degrees(galaxy.get_exploded_supernovae_longitudes())
    num_sn = len(exploded_sn_long)
    logging.info(f"Number of supernovae: {num_sn}")
    # create bin edges for the binning
    bin_edges_long = np.arange(0, 362.5, 2.5) # will end at 360. No data is left out. 145 bin edges
    hist, _ = np.histogram(exploded_sn_long, bins=bin_edges_long) # if a longitude is in the bin, add the intensity to the bin
    # Rearange data to be plotted in desired format
    rearanged_hist = rearange_data(hist) / num_sn
    # Create bin_edges for the plot 
    bin_edges_central = np.arange(2.5, 360, 5)
    bin_edges = np.concatenate(([0], bin_edges_central, [360]))
    # Plot using stairs
    plt.stairs(values=rearanged_hist, edges=bin_edges, fill=False, color='black')
    # Set up the plot
    plt.xlabel("Galactic longitude l (degrees)")
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)

    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ylabel("P(SN)")
    plt.title("Probability density function of SNPs as function of longitude")   
    plt.text(0.02, 0.95, fr'Number of associations: {galaxy.num_asc}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.90, fr'Total number of supernovae progenitors: {num_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.ylim(0, max(rearanged_hist)*1.2) # set the y axis limits
    plt.xlim(0, 360) # so that the plot starts at 0 and ends at 360
    plt.savefig("output/galaxy_tests/sn_as_func_of_long.png", dpi=1200)     # save plot in the output folder
    plt.close()

def plot_mass_distr(galaxy):
    masses = galaxy.get_exploded_supernovae_masses()
    number_sn = np.sum(vectorized_number_sn(galaxy.galaxy))
    mass_max = int(np.ceil(max(masses))) # minimum number of stars = 0
    mass_min = int(np.floor(min(masses)))
    counts, _ = np.histogram(masses, bins=range(mass_min, mass_max + 1, 1))
    plt.plot(range(mass_min, mass_max, 1), counts/np.sum(counts))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, mass_max + 30) # set the x axis limits
    #plt.ylim(0, 1) # set the y axis limits
    plt.xlabel("Mass of SN progenitor (M$_\odot$)")
    plt.ylabel("Probability distribution. P(M$_\odot$)")
    plt.title("Probability distribution for the mass of SN progenitors")
    plt.text(0.02, 0.95, fr'Number of associations: {galaxy.num_asc}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.90, fr'Total number of supernovae progenitors: {number_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.ylim(top=max(counts/np.sum(counts))*3) # set the y axis limits
    plt.savefig("output/galaxy_tests/sn_mass_distribution.png", dpi=1200)     # save plot in the output folder
    plt.close()


def plot_draw_positions_rad_long_lat(galaxy):
    # would be interesting to plot this together with the spiral arm medians
    xs = []
    ys = []
    for asc_number in range(galaxy.num_asc):
        xs.append(galaxy.galaxy[asc_number].x)
        ys.append(galaxy.galaxy[asc_number].y)
    plt.scatter(xs, ys, s=1, color='black')
    plt.plot(0, 0, 'o', color='blue', markersize=10, label='Centre of Galaxy')
    plt.plot(0, r_s, 'o', color='red', markersize=5, label='Centre of Sun')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Distance from the Galactic Centre (kpc)")
    plt.ylabel("Distance from the Galactic Centre (kpc)")
    plt.suptitle("Associations generated from the model.")
    plt.title(f"Made with {galaxy.num_asc} associations")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig("output/galaxy_tests/positions_from_density_distribution.png", dpi=1200)     # save plot in the output folder
    plt.close()


def plot_association(association, creation_time, simulation_time):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    association.plot_association(ax)
    ax.set_xlabel('X (pc from Association Center)')
    ax.set_ylabel('Y (pc from Association Center)')
    ax.set_zlabel('Z (pc from Association Center)')
    plt.suptitle(f"Association created {creation_time} Myr ago. Position of Supernovaes {simulation_time} Myr ago.")
    plt.title(f"Position of Association centre in xyz-coordinates (kpc): ({association.x[0]:.2f}, {association.y[0]:.2f}, {association.z[0]:.2f})")
    legend_exploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=4, label='Exploded')
    legend_unexploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=3, label='Not Exploded')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([legend_exploded, legend_unexploded])
    plt.legend(handles=handles)
    plt.show()
    plt.close()


def plot_association_3d(ax, association, creation_time, simulation_time):
    association.plot_association(ax)
    ax.set_xlabel('X (pc from AC.)')
    ax.set_ylabel('Y (pc from AC.)')
    ax.set_zlabel('Z (pc from AC.)')
    ax.set_title(f"Position of Supernovaes {simulation_time} Myr ago.")

       
def plot_diffusion_of_sns_3d():
    # Here, I only want to look at 1 specific association to see how the SN's diffuse away from the association centre
    creation_time = 40 # Myrs ago
    test_ass = ass.Association(1, creation_time, 20)
    
    # Create a figure
    fig = plt.figure(figsize=(11, 10))
    # First subplot - 3D Surface Plot
    ax1 = fig.add_subplot(221, projection='3d')
    plot_association_3d(ax1, test_ass, creation_time, simulation_time=40)

    test_ass.update_sn(20)
    ax2 = fig.add_subplot(222, projection='3d')
    plot_association_3d(ax2, test_ass, creation_time, 20)

    test_ass.update_sn(10)
    ax3 = fig.add_subplot(223, projection='3d')
    plot_association_3d(ax3, test_ass, creation_time, 10)

    test_ass.update_sn(1)
    ax4 = fig.add_subplot(224, projection='3d')
    plot_association_3d(ax4, test_ass, creation_time, 1)
    
    legend_exploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=4, label='Exploded')
    legend_unexploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=3, label='Not Exploded')
    legend_centre = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Association centre (AC)')
    handles = [legend_centre, legend_exploded, legend_unexploded]
    fig.legend(handles=handles)
    plt.suptitle(f"Position of Association centre in xyz-coordinates (kpc): ({test_ass.x[0]:.2f}, {test_ass.y[0]:.2f}, {test_ass.z[0]:.2f}). \n Association created {creation_time} Myr ago. ")
    plt.savefig("output/galaxy_tests/plot_diffusion_of_sns.png", dpi=1200)


def plot_diffusion_of_sns():
    # Here, I only want to look at 1 specific association to see how the SN's diffuse away from the association centre
    creation_time = 40 # Myrs ago
    test_ass = ass.Association(1, creation_time, 20)
    plot_association(test_ass, creation_time, simulation_time=40)
    test_ass.update_sn(20)
    plot_association(test_ass, creation_time, 20)
    test_ass.update_sn(10)
    plot_association(test_ass, creation_time, 10)
    test_ass.update_sn(1)
    plot_association(test_ass, creation_time, 1)
   
    
def plot_age_mass_distribution():
    tau_0 = 1.6e8 * 1.65
    beta = -0.932 #original
    mass = np.arange(8, 120.1, 0.1)
    time_of_death = tau_0 * (mass)**(beta)
    plt.plot(mass, time_of_death, zorder=0)
    plt.title("Lifetime as function of stellar mass")
    plt.xlabel("Mass of SN progenitor (M$_\odot$)")
    plt.ylabel("Lifetime (yrs)")
    x_vals = [8, 20, 40, 60, 80, 100, 120]
    y_val_index = [0, 120, 320, 520, 720, 920, 1120]
    for i, x in enumerate(x_vals):
        y = time_of_death[y_val_index[i]]
        plt.scatter(x, y, s=30, label=f"{x} M$_\odot$, f(M) = {y:.2e} yrs", zorder=1)
    plt.legend()
    plt.savefig("output/galaxy_tests/age_distribution.png", dpi=1200)  # save plot in the output folder
    plt.show()
    plt.close()


def test_association_placement():
    # temp function: just for testing association placement more quickly to find correct code
    # observation: using interpolated_densities = np.load('output/galaxy_data/interpolated_densities.npy') is ABSOLUTELLY TRASH
    num_lats = len(np.lib.format.open_memmap('output/galaxy_data/latitudes.npy'))
    num_rads = len(np.lib.format.open_memmap('output/galaxy_data/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap('output/galaxy_data/longitudes.npy'))
    # positions:
    x_grid = np.lib.format.open_memmap('output/galaxy_data/x_grid.npy')
    y_grid = np.lib.format.open_memmap('output/galaxy_data/y_grid.npy')
    z_grid = np.lib.format.open_memmap('output/galaxy_data/z_grid.npy')
    # densities:
    emissivity_longitudinal = np.load('output/galaxy_data/emissivity_longitudinal.npy')
    emissivity_longitudinal = emissivity_longitudinal/np.sum(emissivity_longitudinal) # normalize to unity
    emissivity_lat = np.load('output/galaxy_data/emissivity_long_lat.npy')
    emissivity_lat = emissivity_lat/np.sum(emissivity_lat, axis=1, keepdims=True) # normalize to unity for each latitude
    emissivitty_rad = np.load('output/galaxy_data/emissivity_rad_long_lat.npy')
    emissivitty_rad = emissivitty_rad/np.sum(emissivitty_rad, axis=0, keepdims=True) # normalize to unity for each radius
    
    rng = np.random.default_rng()
    NUM_ASC = 10000
    for _ in range(NUM_ASC):
        long_index = rng.choice(a=len(emissivity_longitudinal), size=1, p=emissivity_longitudinal )
        lat_index = rng.choice(a=len(emissivity_lat[long_index].ravel()), size=1, p=emissivity_lat[long_index].ravel() )
        radial_index = rng.choice(a=len(emissivitty_rad[:,long_index,lat_index].ravel()), size=1, p=emissivitty_rad[:, long_index, lat_index].ravel() )
        grid_index = radial_index * num_longs * num_lats + long_index * num_lats + lat_index # 1800 = length of longitudes, 21 = length of latitudes
        x = x_grid[grid_index]
        y = y_grid[grid_index]
        #z = z_grid[grid_index] # not needed
        plt.plot(x, y, 'o', color='black', markersize=1)
    plt.plot(0, 0, 'o', color='blue', markersize=10, label='Centre of Galaxy')
    plt.plot(0, r_s, 'o', color='red', markersize=5, label='Centre of Sun')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Distance from the Galactic Centre (kpc)")
    plt.ylabel("Distance from the Galactic Centre (kpc)")
    plt.suptitle("Associations drawn from the NII density distribution of the Milky Way")
    plt.title(f"Made with {NUM_ASC} associations")
    plt.legend()
    plt.savefig("output/galaxy_tests/test_association_placement.png", dpi=1200)  # save plot in the output folder
    plt.close()
 
def test_plot_density_distribution():
    # let's make a plot for the density distribution of the Milky Way, to see if these maps actually reproduce the expected desnity distribution
    NUM_INTERPOLANT_FILES = 4
    total_galactic_density_unweighted = np.load(os.path.join(WORK_DIRECTORY, 'galaxy_data/interpolated_arm_0.npy')) 
    for i in range(NUM_INTERPOLANT_FILES - 1):
        total_galactic_density_unweighted += np.lib.format.open_memmap(os.path.join(WORK_DIRECTORY, f'galaxy_data/interpolated_arm_{i+1}.npy'))
    #np.load('output/galaxy_data/total_galactic_density_unweighted.npy')
    num_lats = len(np.lib.format.open_memmap('output/galaxy_data/latitudes.npy'))
    num_rads = len(np.lib.format.open_memmap('output/galaxy_data/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap('output/galaxy_data/longitudes.npy'))
    print("Loaded the data")
    total_galactic_density_unweighted = np.reshape(total_galactic_density_unweighted, (num_rads, num_longs, num_lats))
    total_galactic_density_unweighted = total_galactic_density_unweighted[:,:,0]
    total_galactic_density_unweighted = total_galactic_density_unweighted.ravel()
    x_grid = np.load('output/galaxy_data/x_grid.npy')
    y_grid = np.load('output/galaxy_data/y_grid.npy')
    x_grid = np.reshape(x_grid, (num_rads, num_longs, num_lats))
    y_grid = np.reshape(y_grid, (num_rads, num_longs, num_lats))
    x_grid = x_grid[:,:,0]
    y_grid = y_grid[:,:,0]
    x_grid = x_grid.ravel()
    y_grid = y_grid.ravel()
    # Plot the unweigthed density distribution:
    print("Beginning to plot the figure")
    plt.scatter(x_grid, y_grid, c=total_galactic_density_unweighted, cmap='viridis', s=1) 
    plt.scatter(0, 0, c = 'magenta', s=2, label='Galactic centre')
    plt.scatter(0, r_s, c = 'gold', s=2, label='Sun')
    plt.gca().set_aspect('equal')
    plt.xlabel('Distance in kpc from the Galactic center')
    plt.ylabel('Distance in kpc from the Galactic center')
    plt.suptitle('Heatmap of the unweighted interpolated densities of spiral arms in our model')
    plt.title('Normalized to the maximum value')
    plt.legend(loc='upper right')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    filename_unweighted = 'galaxy_tests/test_plot_density_distribution_unweighted_new_high_res.png'
    filepath = os.path.join(WORK_DIRECTORY, filename_unweighted)
    print("Beginning to save the figure")
    plt.savefig(filepath, dpi=1200)  # save plot in the output folder
    plt.close()
    print("Done saving the figure")

    del total_galactic_density_unweighted
    gc.collect()

    # Plot the weigthed density distribution:
    total_galactic_density_weighted = np.load(os.path.join(WORK_DIRECTORY, 'galaxy_data/interpolated_arm_emissivity_0.npy')) 
    for i in range(NUM_INTERPOLANT_FILES - 1):
        total_galactic_density_weighted += np.lib.format.open_memmap(os.path.join(WORK_DIRECTORY, f'galaxy_data/interpolated_arm_emissivity_{i+1}.npy')) 
    #total_galactic_density_weighted = np.load('output/galaxy_data/total_galactic_density_weighted.npy')
    print("Loaded the data")
    total_galactic_density_weighted = np.reshape(total_galactic_density_weighted, (num_rads, num_longs, num_lats))
    total_galactic_density_weighted = np.sum(total_galactic_density_weighted, axis=2)
    total_galactic_density_weighted = total_galactic_density_weighted.ravel()
    total_galactic_density_weighted /= np.max(total_galactic_density_weighted) # normalize this so that the maximum value is 1
    
    print("Shape interpolated_densities: ", total_galactic_density_weighted.shape)
    print("Shape x_grid: ", x_grid.shape)
    print("Shape y_grid: ", y_grid.shape)
    plt.scatter(x_grid, y_grid, s=1, c=total_galactic_density_weighted, cmap='viridis')
    plt.scatter(0, 0, c = 'magenta', s=2, label='Galactic centre')
    plt.scatter(0, r_s, c = 'gold', s=2, label='Sun')
    plt.gca().set_aspect('equal')
    plt.xlabel('Distance in kpc from the Galactic center')
    plt.ylabel('Distance in kpc from the Galactic center')
    plt.suptitle('Heatmap of the weighted interpolated densities of spiral arms in our model')
    plt.title('Normalized to the maximum value')
    plt.legend(loc='upper right')
    cbar = plt.colorbar()
    cbar.set_label('Density')
    print("Beginning to save the figure")
    filename_weighted = 'galaxy_tests/test_plot_density_distribution_weigthed_new_high_res.png'
    filepath = os.path.join(WORK_DIRECTORY, filename_weighted)
    plt.savefig(filepath, dpi=1200)  # save plot in the output folder
    plt.close()
    print("Done saving the figure")
    
    
def run_tests(C, T):
    #test_association_placement()

    galaxy_1 = galaxy.Galaxy(T, star_formation_episodes=1) # an array with n associations
    
    # plot for the cumulative cluster distribution with temporal clustering:
    # plot_age_mass_distribution()
    # plot_diffusion_of_sns_3d()
    print("Running plot_sn_as_func_of_long:")
    plot_sn_as_func_of_long(galaxy_1)
    """ print("Running plot_mass_distr:")
    plot_mass_distr(galaxy_1)
    print("Running plot_draw_positions_rad_long_lat:")
    plot_draw_positions_rad_long_lat(galaxy_1)
    print("Running test_association_placement:")
    test_association_placement() """
    """ galaxy_2 = galaxy.Galaxy(int(np.round(T)), star_formation_episodes=3) # an array with n associations
    galaxy_3 = galaxy.Galaxy(int(np.round(T)), star_formation_episodes=5) # an array with n associations
    ass_models = np.array([galaxy_1, galaxy_2, galaxy_3])
    print("Running plot_cum_snp_cluster_distr")
    plot_cum_snp_cluster_distr(ass_models, C) """
    
def main():
    # other levels for future reference: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
    logging.basicConfig(level=logging.INFO) 
    run_tests(C=C, T=150)

    #plot_diffusion_of_sns_3d()savefig
    #test_plot_density_distribution()

if __name__ == "__main__":
    test_plot_density_distribution()
    #main()