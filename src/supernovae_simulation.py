import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

rng = np.random.default_rng()

""" galactic_densities = np.loadtxt('output\long_lat_skymap.txt')
galactic_densities_as_long = np.sum(galactic_densities[1:, 1:], axis=1) """
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes



def plot_intensity(galactic_intensities_long):
    plt.plot(np.linspace(0, 100, len(galactic_intensities_long)), galactic_intensities_long)
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Galactic Longitude (degrees)')
    plt.ylabel('Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
    plt.show()
    plt.close()


def monteCarlo_hit_or_miss(x_range, function_values):
    # implements the hit or miss method
    f_max = np.max(function_values)
    r = rng.random()
    x = np.round(r * x_range[-1])
    if function_values[int(x)] > f_max * rng.random():
        return x
    else:
        return monteCarlo_hit_or_miss(x_range, function_values)


def test_monteCarlo_draw_bubbles(galactic_densities_as_long):
    x = []
    x_range = np.arange(0, len(galactic_densities_as_long), 1)
    for _ in range(100000):
        x.append(monteCarlo_hit_or_miss(x_range, galactic_densities_as_long))
    plt.hist(np.array(x), bins=100, density=True)
    plt.show()
    plt.close()
# plot_intensity(galactic_densities_as_long)

def test_np_choise(galactic_densities_as_long):
    x = np.random.choice(np.arange(0, len(galactic_densities_as_long), 100000), size=1, p=galactic_densities_as_long/np.sum(galactic_densities_as_long))
    plt.hist(x, bins=100, density=True)
    plt.show()
    plt.close()


def monteCarlo_temporal(c):
    # c = number of star formation episodes
    r= rng.random() # random number between 0 and 1. 1 is excluded
    return np.exp((c - r)/0.11)


def imf_kroupa():
    solar_masses = np.arange(0.01, 120, 0.01) # mass in solar masses
    m = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
    alpha = np.array([0.3, 1.3, 2.3, 2.7])
    k = 0.877
    n = []
    for mass in solar_masses:
        if mass < 0.08: # n = 0
            n.append((mass/m[1])**(-alpha[0]))
        elif mass < 0.5: # n = 1
            n.append((mass/m[1])**(-alpha[1]))
        elif mass < 1: # n = 2
            n.append((m[2]/m[1])**(-alpha[1]) * (mass/m[2])**(-alpha[2]))
        else: # n = 3
            n.append((m[2]/m[1])**(-alpha[1]) * (m[3]/m[2])**(-alpha[2]) * (mass/m[3])**(-alpha[3]))
    n = np.array(n)
    return solar_masses, n/np.sum(n)


def test_imf():
    solar_masses, n = imf_kroupa()
    solar_masses_salpeter = np.arange(0.1, 120, 0.01) # mass in solar masses
    salpeter = (solar_masses_salpeter)**(-2.35) 
    plt.plot(solar_masses, n, label='Kroupa IMF')
    plt.plot(solar_masses_salpeter, salpeter/np.sum(salpeter*0.01), label='Salpeter IMF')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mass (M$_\odot$)')
    plt.ylabel('Number of stars')
    plt.title('Kroupa IMF vs Salpeter IMF. Each curve is normalized to unity')
    plt.legend()
    plt.show()

test_imf()

# x,y,z coordinates for the OB associations and their stars
# give them a velcoity, isotropic distribution
# keep those who have exploded within the allowed timeframe. Model shall have x years as input (i.e. "I want to know about the SN's that exploded in the last x years")
# For the Gum-Sygnus: give them value one within the sphere, zero outside. Implement a sort of running-average, where the value along our line of sight is the average of the values in the sphere weighted by a gaussian distribution to simulate the 7 degree FIRAS beam


def test_mc_imf():
    # draw random number between 0 and 1:
    solar_masses = np.arange(8, 120, 0.01) # mass in solar masses
    m = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
    alpha = np.array([0.3, 1.3, 2.3, 2.7])
    imf = (m[2]/m[1])**(-alpha[1]) * (m[3]/m[2])**(-alpha[2]) * (solar_masses/m[3])**(-alpha[3])
    m = np.random.choice(solar_masses, size=1, p=imf/np.sum(imf))
    #m = monteCarlo_hit_or_miss(solar_masses - 8, imf/np.sum(imf))
    return m

    
def test_plot_mc_imf():
    x = []
    for _ in range(100000):
        x.append(test_mc_imf() + 8)
    plt.hist(np.array(x), bins=100, density=True)
    plt.xscale('log')
    plt.xlabel('Mass (M$_\odot$)')
    plt.ylabel('Number of stars')
    plt.title('Monte Carlo simulation of the IMF')
    plt.legend()
    plt.yscale('log')
    plt.show()
    plt.close()

#test_plot_mc_imf()
