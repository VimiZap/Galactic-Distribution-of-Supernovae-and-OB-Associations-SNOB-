import numpy as np
import matplotlib.pyplot as plt
import supernovae_class as sn
import association_class as ass
from matplotlib.ticker import AutoMinorLocator


N = 10e4 # number of associations in the Galaxy
T = 20 # simulation run time in Myrs
star_formation_episodes = [1, 3, 5]
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes


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

def plot_cum_snp_cluster_distr(association_array, n, C):
    # P(<N*): the cumultative distribution of steller clusters as function of number of snp's
    # n = number of associations
    # c = array with the value of c 
    for i in range(len(C)):
        print('i = ', i)
        association_array_num_sn = vectorized_number_sn(association_array[i])
        num_bins = int(np.ceil(max(association_array_num_sn))) # minimum number of stars = 0
        counts, _ = np.histogram(association_array_num_sn, bins=range(0, num_bins, 1))
        cumulative = (n - np.cumsum(counts))/n # cumulative distribution, normalized
        plt.plot(range(1, num_bins, 1), cumulative, label="Number of star formation episodes = " + str(C[i]))
    plt.xscale("log")
    plt.xlim(1, num_bins + 3000) # set the x axis limits
    plt.ylim(0, 1) # set the y axis limits
    plt.xlabel("Number of SNPs")
    plt.ylabel("Cumulative distribution. P(N > x)")
    plt.suptitle("Monte Carlo simulation of temporal clustering of SNPs")
    plt.title(f"Made with {n} associations")
    plt.legend()
    plt.savefig("output/galaxy_tests/temporal_clustering.png", dpi=1200)     # save plot in the output folder
    #plt.show()


def plot_sn_as_func_of_long(association_array, n):
    longitudes = np.array([])
    number_sn = np.array([])
    for i in range(len(association_array)):
        longitudes = np.concatenate((longitudes, association_array[i].longitudes.ravel()))
        number_sn += np.sum(vectorized_number_sn(association_array[i]))
    longitudes_sn = np.sort(longitudes)
    dl = 2   # increments in dl (degrees):
    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))

    counts, bins = np.histogram(longitudes_sn, bins=len(longitudes))
    counts_l1 = counts[len(l1)-1::-1] # first half
    counts_l2 = counts[-1:len(l1)-1:-1] # second half
    x_values = np.linspace(0, len(longitudes), len(longitudes))
    print("len l1: ", len(l1))
    print("len l2: ", len(l2))
    print("length counts: ", len(counts))
    print("length counts_l1: ", len(counts_l1))
    print("length counts_l2: ", len(counts_l2))
    plt.plot(x_values, np.concatenate((counts_l1, counts_l2))/np.sum(counts))
    plt.xlabel("Galactic longitude l (degrees)")

    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, len(longitudes), 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ylabel("P(SN)")
    plt.title("Probability density function of SNPs as function of longitude")   
    plt.text(0.02, 0.95, fr'Number of associations: {n}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.90, fr'Total number of supernovae progenitors: {number_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.savefig("output/galaxy_tests/sn_as_func_of_long.png", dpi=1200)     # save plot in the output folder
    #plt.show()


def plot_mass_distr(association_array, n):
    masses = np.array([])
    number_sn = np.array([])
    for i in range(len(association_array)):
        masses = np.concatenate((masses, association_array[i].find_sn_masses))
        number_sn += np.sum(vectorized_number_sn(association_array[i]))
    mass_max = int(np.ceil(max(masses))) # minimum number of stars = 0
    mass_min = int(np.floor(min(masses)))
    counts, _ = np.histogram(masses, bins=range(mass_min, mass_max + 1, 1))
    plt.plot(range(mass_min, mass_max, 1), counts/np.sum(counts))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, mass_max + 30) # set the x axis limits
    plt.ylim(0, 1) # set the y axis limits
    plt.xlabel("Mass of SN progenitor (M$_\odot$)")
    plt.ylabel("Probability distribution. P(M$_\odot$)")
    plt.title("Probability distribution for the mass of SN progenitors")
    plt.text(0.02, 0.95, fr'Number of associations: {n}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.90, fr'Total number of supernovae progenitors: {number_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.savefig("output/galaxy_tests/sn_mass_distribution.png", dpi=1200)     # save plot in the output folder
    #plt.show()
    

def plot_draw_positions_rad_long_lat(association_array, n):
    # would be interesting to plot this together with the spiral arm medians
    xs = []
    ys = []
    for i in range(len(association_array)):
        xs.append(association_array[i].x)
        ys.append(association_array[i].y)
    plt.scatter(xs, ys, s=1, color='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Distance from the Galactic Centre (kpc)")
    plt.ylabel("Distance from the Galactic Centre (kpc)")
    plt.suptitle("Associations drawn from the NII density distribution of the Milky Way")
    plt.title(f"Made with {n} associations")
    plt.savefig("output/galaxy_tests/positions_from_density_distribution.png", dpi=1200)     # save plot in the output folder
    plt.show()

def run_tests(n, C, T):
    association_array_1 = generate_galaxy(n, C[0], T) # an array with n associations
    association_array_2 = generate_galaxy(n, C[1], T) # an array with n associations
    association_array_3 = generate_galaxy(n, C[2], T) # an array with n associations
    ass_models = np.array([association_array_1, association_array_2, association_array_3])
    # plot for the cumulative cluster distribution with temporal clustering:

    plot_cum_snp_cluster_distr(ass_models, n, C)
    plot_sn_as_func_of_long(association_array_1, n)
    plot_mass_distr(association_array_1, n)
    plot_draw_positions_rad_long_lat(association_array_1, n)


run_tests(10000, C, T)
