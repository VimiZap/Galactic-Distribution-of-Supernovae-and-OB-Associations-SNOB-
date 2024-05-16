import numpy as np
import matplotlib.pyplot as plt
import src.galaxy_model.supernovae_class as sn
import src.utilities.utilities as ut
import src.utilities.constants as const
import src.galaxy_model.galaxy_class as galaxy

#AVG_SN_PER_ASC = np.array([204, 620, 980]) # number of star formation episodes = 1, 3, 5
SN_BIRTHRATE = 2.81e4 # units of SN/Myr
ASC_BIRTHRATE = np.array([3084, 3085, 3085])  # number of associations created per Myr
SNS_BIRTHRATE = np.array([23270, 26600, 28100])  # number of supernovae progenitors created per Myr
AVG_NUM_SNP_PER_ASC = np.array([8, 9, 9]) # small and similar number - should just take 10 for all
STAR_FORMATION_EPISODES = [1, 3, 5]
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes


solar_masses = np.arange(8, 120 + 0.01, 0.01) # mass in solar masses. Used to draw random masses for the SNPs in the association
imf = ut.imf_3(solar_masses) # the imf for the range 1 <= M/M_sun < 120
rng = np.random.default_rng()

def draw_sn_masses(it):
    sn_masses = np.array([])
    for i in range(it):
        n_min = 2
        n_max = 1870 #* self._star_formation_episodes
        num_snp = calc_num_associations(n_min, n_max, c=C[2])
        size = np.sum(num_snp, dtype=int) # total number of SNPs in the association
        #size = rng.integers(1, 1000, endpoint=True, size=1)
        sn_masses = np.concatenate((sn_masses, rng.choice(solar_masses, size=size, p=imf/np.sum(imf)))) # draw random masses for the SNPs in the association from the IMF in the range 8 <= M/M_sun < 120
    return sn_masses


def plot_mass_distr_test():
    # Add the actual Kroupa IMF to the plot
    drawn_masses = draw_sn_masses(100)
    mass, imf = ut.imf() # entire IMF
    m3 = np.arange(8, 120, 0.01) # mass for the last part of the IMF
    imf3 = ut.imf_3(m3) # the imf for the range 8 <= M/M_sun < 120
    imf = imf / np.sum(imf3) / 0.01 # normalize the imf to unity for the mass range 8 <= M/M_sun < 120
    # Modelled data
    """ drawn_masses = np.array([]) # array to store the drawn masses (M/M_sun) for the supernovae progenitors
    number_sn = 0
    for asc in galaxy.associations:
        number_sn += np.sum(asc.number_sn)
        drawn_masses = np.concatenate((drawn_masses, asc.star_masses)) """
    #drawn_masses = drawn_masses.flatten()
    drawn_masses = drawn_masses[drawn_masses >= 8]
    mass_max = int(np.ceil(max(drawn_masses))) 
    mass_min = int(np.floor(min(drawn_masses))) # minimum number of stars = 0
    binwidth = 1
    bins = np.arange(mass_min, mass_max + binwidth, binwidth)
    counts, _ = np.histogram(drawn_masses, bins=bins)
    counts = counts / np.sum(counts) / binwidth 
    plt.figure(figsize=(8, 8))
    plt.plot(bins[:-1], counts, label='Stellar masses in modelled Galaxy', color='blue')
    plt.plot(mass, imf, label='The modified Kroupa IMF', color='black', linestyle='dashed')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(np.min(mass), mass_max + 30) # set the x axis limits
    plt.ylim(bottom = 5e-5, top=1e4) # set the y axis limits
    plt.xlabel("Mass of SN progenitor (M$_\odot$)")
    plt.ylabel("Probability distribution. P(M$_\odot$)")
    plt.legend(loc='lower left')
    plt.savefig('sn_mass_distribution.pdf')     # save plot in the output folder
    plt.close()


def plot_mass_distr(galaxy):
    """ Function to plot the probability distribution for the mass of SNP's. SNP's from a generated galaxy
    
    Args:
        galaxy: The galaxy to plot the SNP mass distribution for
    
    Returns:
        None. Saves a plot in the output folder
    """
    # Add the actual Kroupa IMF to the plot
    mass, imf = ut.imf() # entire IMF
    m3 = np.arange(8, 120, 0.01) # mass for the last part of the IMF
    imf3 = ut.imf_3(m3) # the imf for the range 8 <= M/M_sun < 120
    imf = imf / np.sum(imf3) / 0.01 # normalize the imf to unity for the mass range 8 <= M/M_sun < 120
    # Modelled data
    drawn_masses = np.array([]) # array to store the drawn masses (M/M_sun) for the supernovae progenitors
    number_sn = 0
    for asc in galaxy.associations:
        number_sn += np.sum(asc.number_sn)
        drawn_masses = np.concatenate((drawn_masses, asc.star_masses))
    #drawn_masses = drawn_masses.flatten()
    drawn_masses = drawn_masses[drawn_masses >= 8]
    mass_max = int(np.ceil(max(drawn_masses))) 
    mass_min = int(np.floor(min(drawn_masses))) # minimum number of stars = 0
    binwidth = 1
    bins = np.arange(mass_min, mass_max + binwidth, binwidth)
    counts, _ = np.histogram(drawn_masses, bins=bins)
    counts = counts / np.sum(counts) / binwidth 
    plt.figure(figsize=(8, 8))
    plt.plot(bins[:-1], counts, label='Stellar masses in modelled Galaxy', color='blue')
    plt.plot(mass, imf, label='The modified Kroupa IMF', color='black', linestyle='dashed')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(np.min(mass), mass_max + 30) # set the x axis limits
    plt.ylim(bottom = 5e-5, top=1e4) # set the y axis limits
    plt.xlabel("Mass of SN progenitor (M$_\odot$)")
    plt.ylabel("Probability distribution. P(M$_\odot$)")
    plt.legend(loc='lower left')
    print('saving figure')
    plt.savefig('sn_mass_distribution_galaxy_30.pdf')     # save plot in the output folder
    plt.close()

def association_distribution(n_min, n_max, N = None):
    # it is assumed n_min and n_max are adjusted for the numbner of star forming episodes
    constant = 1.65 * 1.7e6 * 1.1e-3 # 1.1e-3 = f_SN = the fraction of stars that end their lives as core-collapse supernovae
    if N == None: # if N is not given, generate the range of N. N != None is interpreted as the user wants the number of associations for a specific N
        N = np.arange(n_min, n_max + 1) # default step size is 1. The range is inclusive of n_max
        return constant / N**(1.8) ##########################################################################################################################################################################
    return int(np.ceil(constant / N**2))


def association_distribution_normalized(n_min, n_max, c):
    distribution = association_distribution(n_min, n_max)
    return distribution / np.sum(distribution)


def calc_num_associations(n_min, n_max, c):
        N = np.arange(n_min, n_max + 1)
        # Using the normalized association distribution to draw N
        distribution = association_distribution_normalized(n_min, n_max, c)
        #distribution = self._association_distribution_cumulative_distribution(n_min, n_max, c)
        """ plt.plot(N, distribution)
        plt.xscale('log')
        plt.show()
        plt.close()
        plt.plot(N, distribution1)
        plt.xscale('log')
        plt.show()
        plt.close() """
        # Draw the actual number of associations as given by a random multinomial distribution
        #num_snp_drawn = self.rng.choice(a=N, size=1000, p=distribution)
        num_snp_drawn = []
        num_snp_target = SNS_BIRTHRATE[2]
        count = 0
        #print(f'number of star forming episodes: {self._star_formation_episodes}. n_min, n_max: {n_min, n_max}. num_snp_target: {num_snp_target}.')
        min_num_snp = np.inf
        while np.sum(num_snp_drawn) < num_snp_target*0.99:
            count += 1
            new_num_snp_drawn = rng.choice(a=N, size=1, p=distribution)
            new_num_snp_drawn = np.ones(5) * new_num_snp_drawn # As Mckee and Williams 1997 finds, the star forming episodes are of equal size most probably
            #new_num_snp_drawn = np.sum(new_num_snp_drawn, dtype=int)
            #new_num_snp_drawn = int(new_num_snp_drawn)
            if np.sum(new_num_snp_drawn) < n_min*5:
                print('WTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
            if np.sum(new_num_snp_drawn) < min_num_snp:
                min_num_snp = np.sum(new_num_snp_drawn)
            if np.sum(new_num_snp_drawn) <= 2 & 5 > 1:
                raise ValueError('new_num_snp_drawn <= 2')
            #print('new_num_snp_drawn:', new_num_snp_drawn)
            #new_num_snp_drawn *= self._star_formation_episodes
            #print('new_num_snp_drawn:', new_num_snp_drawn)
            """ diff = num_snp_target - np.sum(num_snp_drawn) - new_num_snp_drawn
            if diff < 0: 
                print('diff:', diff)
                print('count:', count) """
                #new_num_snp_drawn = new_num_snp_drawn + diff
            #num_snp_drawn = np.concatenate((num_snp_drawn, [new_num_snp_drawn]))
            num_snp_drawn.append(new_num_snp_drawn)
            if count <= 10:
                a=0
                #print(f'num_snp_drawn: {num_snp_drawn}. np.sum(num_snp_drawn): {np.sum(num_snp_drawn)}.')
        print(f'count: {count}. num star formation episodes: {5}. min_num_snp: {min_num_snp}. np.min(num_snp_drawn): {np.min(num_snp_drawn)}. np.max(num_snp_drawn): {np.max(num_snp_drawn)}.')
        return num_snp_drawn


plot_mass_distr_test()
galaxy1 = galaxy.Galaxy(sim_time_duration=30, star_formation_episodes=5)
plot_mass_distr(galaxy1)