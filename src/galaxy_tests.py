import numpy as np
import matplotlib.pyplot as plt
import supernovae_class as sn
import association_class as ass
import galaxy_class as galaxy
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D


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
    # Sum along the specified axis (axis=1 sums along columns)
    result = np.sum(paired_data, axis=1)
    return result

def plot_sn_as_func_of_long(galaxy):
    longitudes = np.array([])
    
    print("len association_array: ", galaxy.num_asc)
    longitudes = galaxy.get_exploded_supernovae_longitudes()
    number_sn = len(longitudes)
    print("Number_sn: ", number_sn)
    longitudes_sn = np.sort(longitudes)
    dl = 1   # increments in dl (degrees):
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))
    counts, bins = np.histogram(longitudes_sn, bins=len(longitudes))
    """ bin_edges = np.arange(0, 362.5, 2.5)
    bin_width = 5
    
    hist, bin_edges = np.histogram(longitudes_sn, bins=bin_edges) # if a longitude is in the bin, add the intensity to the bin
    
    x_1 = 0 # 180 degrees
    x_2 = np.arange(2.5, 360 - 2.5, 5) # 180-2.5, down to 0, up to 180 degrees again
    x_3 = 360 - 2.5 # 180 degrees
    print("x2 length: ", len(x_2))
    print("x1: ", x_1)
    print("x2: ", x_2)
    print("x3: ", x_3)
    data_centre_left = hist[0]
    data_left = sum_pairwise(hist[1:71]) / 2 
    data_left_edge = hist[71]
    data_right_edge = hist[72]
    data_right = sum_pairwise(hist[73:-1]) / 2
    data_centre_right = hist[-1]

    print(data_left.shape, data_right.shape, data_centre_left.shape)
    print(data_centre_left)
    print(len(data_left), len(data_right))
    print(data_left[-1], data_left_edge)
    print(1+1 + len(data_left) + 1+1 + len(data_right))
    print(len(hist))
    print(len(bin_edges))

    data_central= np.concatenate((data_left[::-1], [(data_centre_left + data_centre_right)/2], data_right[::-1]))
    plt.bar(x_1, data_left_edge, width=bin_width/2, align='edge', edgecolor='black', color='blue')
    plt.bar(x_2, data_central, width=bin_width, align='edge', edgecolor='black', color='blue')
    plt.bar(x_3, data_right_edge, width=bin_width/2, align='edge', edgecolor='black', color='blue')
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.xlabel('Galactic longitude (degrees)')
    plt.ylabel('Line intensity in some units')
    plt.title("N+ line intensity vs Galactic longitude")
    plt.show() """
    counts_l1 = counts[len(l1)-1::-1] # first half
    counts_l2 = counts[-1:len(l1)-1:-1] # second half
    x_values = np.linspace(0, len(longitudes), len(longitudes))
    print("len l1: ", len(l1))
    print("len l2: ", len(l2))
    print("length counts: ", len(counts))
    print("length counts_l1: ", len(counts_l1))
    print("length counts_l2: ", len(counts_l2))
    y_values = np.concatenate((counts_l1, counts_l2))/np.sum(counts)
    plt.plot(x_values, y_values)
    plt.xlabel("Galactic longitude l (degrees)")
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, len(longitudes), 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ylabel("P(SN)")
    plt.title("Probability density function of SNPs as function of longitude")   
    plt.text(0.02, 0.95, fr'Number of associations: {galaxy.num_asc}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.90, fr'Total number of supernovae progenitors: {number_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.ylim(0, max(y_values)*1.2) # set the y axis limits
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
    # a
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
    # positions:
    x_grid = np.load('output/galaxy_data/x_grid.npy')
    y_grid = np.load('output/galaxy_data/y_grid.npy')
    # z_grid = np.load('output/galaxy_data/z_grid.npy') # not needed
    # densities:
    densities_longitudinal = np.load('output\galaxy_data\densities_longitudinal.npy')
    densities_longitudinal = densities_longitudinal/np.sum(densities_longitudinal) # normalize to unity
    densities_lat = np.load('output\galaxy_data\densities_lat.npy')
    densities_lat = densities_lat/np.sum(densities_lat, axis=1, keepdims=True) # normalize to unity for each latitude
    rad_densities = np.load('output\galaxy_data\densities_rad.npy')
    rad_densities = rad_densities/np.sum(rad_densities, axis=0, keepdims=True) # normalize to unity for each radius
    
    rng = np.random.default_rng()
    NUM_ASC = 10000
    for _ in range(NUM_ASC):
        long_index = rng.choice(a=len(densities_longitudinal), size=1, p=densities_longitudinal )
        lat_index = rng.choice(a=len(densities_lat[long_index].ravel()), size=1, p=densities_lat[long_index].ravel() )
        radial_index = rng.choice(a=len(rad_densities[:,long_index,lat_index].ravel()), size=1, p=rad_densities[:, long_index, lat_index].ravel() )
        grid_index = radial_index * 1800 * 21 + long_index * 21 + lat_index # 1800 = length of longitudes, 21 = length of latitudes
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
    total_galactic_density_weighted = np.load('output/galaxy_data/total_galactic_density_weighted.npy')
    total_galactic_density_unweighted = np.load('output/galaxy_data/total_galactic_density_unweighted.npy')
    total_galactic_density_unweighted = np.reshape(total_galactic_density_unweighted, (4818, 1800, 21))
    total_galactic_density_unweighted = total_galactic_density_unweighted[:,:,0]
    total_galactic_density_unweighted = total_galactic_density_unweighted.ravel()
    x_grid = np.load('output/galaxy_data/x_grid.npy')
    y_grid = np.load('output/galaxy_data/y_grid.npy')
    x_grid = np.reshape(x_grid, (4818, 1800, 21))
    y_grid = np.reshape(y_grid, (4818, 1800, 21))
    x_grid = x_grid[:,:,0]
    y_grid = y_grid[:,:,0]
    x_grid = x_grid.ravel()
    y_grid = y_grid.ravel()
    # Plot the unweigthed density distribution:
    plt.scatter(x_grid, y_grid, c=total_galactic_density_unweighted, cmap='viridis', s=1) # MISTAKE HERE: We are not really summing up all the latitudinal contribuitions, just plotting it on top of each other, overlapping
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
    print("Beginning to save the figure")
    plt.savefig("output/galaxy_tests/test_plot_density_distribution_weighted.png", dpi=1200)  # save plot in the output folder
    plt.close()
    print("Done saving the figure")
    
    # Plot the weigthed density distribution:
    total_galactic_density_weighted = np.reshape(total_galactic_density_weighted, (4818, 1800, 21))
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
    plt.savefig("output/galaxy_tests/test_plot_density_distribution_weigthed.png", dpi=1200)  # save plot in the output folder
    plt.close()
    print("Done saving the figure")
    
    
def run_tests(C, T):
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
    

run_tests(C=C, T=100)

#plot_diffusion_of_sns_3d()
#test_plot_density_distribution()