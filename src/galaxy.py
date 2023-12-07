import numpy as np
import matplotlib.pyplot as plt
import supernovae_class as sn
import association_class as ass
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D


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
        num_bins = int(np.ceil(max(association_array_num_sn))) # minimum number of stars = 1
        counts, _ = np.histogram(association_array_num_sn, bins=range(1, num_bins, 1))
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
    plt.close()


def plot_sn_as_func_of_long(association_array, n):
    longitudes = np.array([])
    number_sn = np.sum(vectorized_number_sn(association_array))
    print("Number_sn: ", number_sn)
    print("len association_array: ", len(association_array))
    for i in range(len(association_array)):
        longitudes = np.concatenate((longitudes, association_array[i].longitudes.ravel()))
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
    y_values = np.concatenate((counts_l1, counts_l2))/np.sum(counts)
    plt.plot(x_values, y_values)
    plt.xlabel("Galactic longitude l (degrees)")

    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, len(longitudes), 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ylabel("P(SN)")
    plt.title("Probability density function of SNPs as function of longitude")   
    plt.text(0.02, 0.95, fr'Number of associations: {n}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.90, fr'Total number of supernovae progenitors: {number_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.ylim(0, max(y_values)*1.2) # set the y axis limits
    plt.savefig("output/galaxy_tests/sn_as_func_of_long.png", dpi=1200)     # save plot in the output folder
    plt.close()


def plot_mass_distr(association_array, n):
    masses = np.array([])
    number_sn = np.sum(vectorized_number_sn(association_array))
    for i in range(len(association_array)):
        masses = np.concatenate((masses, association_array[i].find_sn_masses))
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
    plt.text(0.02, 0.95, fr'Number of associations: {n}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.90, fr'Total number of supernovae progenitors: {number_sn}', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.ylim(top=max(counts/np.sum(counts))*3) # set the y axis limits
    plt.savefig("output/galaxy_tests/sn_mass_distribution.png", dpi=1200)     # save plot in the output folder
    plt.close()

    

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
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    legend_exploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=4, label='Exploded')
    legend_unexploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=3, label='Not Exploded')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([legend_exploded, legend_unexploded])
    plt.legend(handles=handles)
    plt.show()
    plt.close()

def plot_diffusion_of_sns():
    creation_time = 40 # Myrs ago
    test_ass = ass.Association(1, creation_time, 20)
    test_ass.print_association()
    plot_association(test_ass, creation_time, simulation_time=40)

    test_ass.update_sn(20)
    test_ass.print_association()
    plot_association(test_ass, creation_time, 20)

    test_ass.update_sn(10)
    test_ass.print_association()
    plot_association(test_ass, creation_time, 10)

    test_ass.update_sn(5)
    test_ass.print_association()
    plot_association(test_ass, creation_time, 5)

    test_ass.update_sn(1)
    test_ass.print_association()
    """ t = sn.simulation_run_time
    age = sn.age
    mass = sn.mass
    exploded = sn.exploded
    print(t, age, mass, exploded) """
    plot_association(test_ass, creation_time, 1)
    

""" def plot_age_mass_distribution():
    tau_0 = 1.6e8 * 1.65
    beta = -0.932
    mass = np.arange(8, 120.1, 0.1)
    time_of_death = tau_0 * (mass)**(beta)
    plt.plot(mass, time_of_death)
    plt.title("Lifetime as function of stellar mass")
    plt.xlabel("Mass of SN progenitor (M$_\odot$)")
    plt.ylabel("Age (yrs)")
    x_vals = np.arange(0, 120 + 1, 20)
    print(x_vals)
    for i, x in enumerate(x_vals):
        y = time_of_death[i]
        # for your last 2 points only
        plt.scatter(x, y, s=10, c='blue')
        if i >= len(x_vals) - 2:
            plt.text(x-.2, y-1, f"({x} M$_\odot$, lifetime = {y:.2f})", horizontalalignment="right", rotation=0)
        # for all other points
        else:
            plt.text(x+.2, y-1, "({x} M$_\odot$, lifetime = {y:.2f})", horizontalalignment="left", rotation=0)
    plt.savefig("output/galaxy_tests/age_distribution.png", dpi=1200)     # save plot in the output folder
    plt.show()
    plt.close() """

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
        """ if i == 0 or i == 1:
            plt.text(x + 2, y, f"({x} M$_\odot$, f(M) = {y:.2e} yrs)", horizontalalignment="left", rotation=0)
        elif i == 2:
            plt.text(x - 5, y + 0.15e7, f"({x} M$_\odot$, f(M) = {y:.2e} yrs)", horizontalalignment="left", rotation=0)
        elif i == 3:
            plt.text(x - 10, y + 0.15e7, f"({x} M$_\odot$, f(M) = {y:.2e} yrs)", horizontalalignment="left", rotation=0)

        else:
            plt.text(x - 10, y + 0.1e7, f"({x} M$_\odot$, f(M) = {y:.2e} yrs)", horizontalalignment="left", rotation=0) """
    plt.legend()
    plt.savefig("output/galaxy_tests/age_distribution.png", dpi=1200)  # save plot in the output folder
    plt.show()
    plt.close()


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

#plot_age_mass_distribution()
plot_diffusion_of_sns()
#run_tests(10000, C, T)
