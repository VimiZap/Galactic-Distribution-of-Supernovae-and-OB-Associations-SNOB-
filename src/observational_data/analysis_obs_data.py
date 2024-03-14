import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import obs_utilities as obs_ut
from scipy.optimize import curve_fit
import src.utilities.utilities as ut
import src.spiral_arm_model as sam
import src.utilities.constants as const
import src.galaxy_model.galaxy_class as gal
import src.galaxy_model.association_class as asc
import logging
logging.basicConfig(level=logging.INFO)
rng = np.random.default_rng()

def my_data():
    file_path = f'{const.FOLDER_OBSERVATIONAL_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    return data


def plot_age_hist(age_data, filename):
    """ Plot the age vs. distance of OB associations
    
    Args:
        age_data: array. Age of the associations
    
    Returns:
        None. Shows the plot
    """
    binwidth = 2
    bins = np.arange(0, 50 + binwidth, binwidth)
    plt.figure(figsize=(10, 6))
    plt.hist(age_data, bins=bins, color='green', edgecolor='black', zorder=10)
    plt.title('Histogram of ages of OB associations')
    plt.xlabel('Age (Myr)')
    plt.xlim(0, 50)
    plt.ylabel('Counts')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # Set the y-axis to only use integer ticks
    plt.grid(axis='y')
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()


def add_heliocentric_circles_to_ax(ax, step=0.5):
    """ Add heliocentric circles to the plot
    
    Args:
        ax: axis. The axis to add the circles to
        step: float. Step size for the radial binning of associations in kpc
    
    Returns:
        None
    """
    thetas_heliocentric_circles = np.linspace(0, 2 * np.pi, 100)
    for i in range(1, 10):
        x_heliocentric_circles = i * step * np.cos(thetas_heliocentric_circles)
        y_heliocentric_circles = i * step * np.sin(thetas_heliocentric_circles) + const.r_s
        ax.plot(x_heliocentric_circles, y_heliocentric_circles, color='black', linestyle='--', linewidth=0.5, zorder=5) # plot the heliocentric circles
    return


def add_spiral_arms_to_ax(ax):
    """ Add the spiral arms to the plot
    
    Args:
        ax: axis. The axis to add the spiral arms to
    
    Returns:
        None
    """
    for i in range(len(const.arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(const.arm_angles[i], const.pitch_angles[i])
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        ax.plot(x, y, color='black', marker='o', linewidth = 0.0001, zorder=6, markeredgewidth=0.0001, markersize=0.0001) # plot the spiral arm medians
    return


def add_associations_to_ax(ax, x, y, label, color):
    """ Add the associations to the plot
    
    Args:
        ax: axis. The axis to add the associations to
        x: array. x-coordinates of the associations. Units of kpc
        y: array. y-coordinates of the associations. Units of kpc
        label: str. Label name for the plotted associations
        color: str. Colour of the plotted associations

    Returns:
        None
    """
    ax.scatter(x, y, color=color, alpha=0.5, s=8, label=label, zorder=10)
    return


def plot_associations(x, y, filename, label_plotted_asc, step=0.5):
    """ Plot the distribution of known associations in the Galactic plane together with the spiral arms medians
    
    Args:
        x: array. x-coordinates of the associations. Units of kpc
        y: array. y-coordinates of the associations. Units of kpc
        filename: str. Name of the file to save the plot
        label_plotted_asc: str. Label name for the plotted associations
        step: float. Step size for the radial binning of associations in kpc
    
    Returns:
        None. Saves the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    add_associations_to_ax(ax, x, y, label=label_plotted_asc, color='blue')
    add_heliocentric_circles_to_ax(ax, step=step)
    add_spiral_arms_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10, zorder=11)
    plt.title('Distribution of associations in the Galactic plane')
    plt.xlabel('x (kpc)')
    plt.ylabel('y (kpc)')
    plt.xlim(-7.5, 7.5)
    plt.ylim(2, 12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()


def area_per_bin(bins):
    """ Calculate the area of each bin in a histogram for a circular bins
    
    Args:
        bins: array. The bins of the histogram
    
    Returns:
        area_per_circle: array. The area of each bin
    """
    area_per_circle = np.power(bins[1:], 2) * np.pi - np.power(bins[:-1], 2) * np.pi
    return area_per_circle


def exponential_falloff(x, a, b, c):
    """
    Exponential falloff function.

    Parameters:
    x : array-like
        Independent variable.
    a : float
        Initial amplitude.
    b : float
        Decay rate.
    c : float
        Constant offset.

    Returns:
    y : array-like
        Dependent variable, representing the exponential falloff.
    """
    return a * np.exp(-b * x) + c


def plot_distance_hist(heliocentric_distance, filename, step=0.5, endpoint=5, fit_exp=True):
    """ Plot the histogram of distances of OB associations and fit the data to a Gaussian function

    Args:
        heliocentric_distance: array. Heliocentric distances of the associations in kpc
        filename: str. Name of the file to save the plot
        step: float. Step size for the histogram in kpc
        endpoint: float. Max radial distance in kpc
        fit_exp: bool. If True, fits the data to an exponential function
    
    Returns:
        None. Saves the plot
    """
    bins = np.arange(0, endpoint + step, step)
    area_per_circle = area_per_bin(bins)
    hist, _ = np.histogram(heliocentric_distance, bins=bins)
    hist = hist / area_per_circle # find the surface density of OB associations
    hist_central_x_val = bins[:-1] + step / 2 # central x values for each bin
    # Make the histogram    
    plt.figure(figsize=(10, 6))
    plt.bar(hist_central_x_val, hist, width=step, color='green', alpha=0.7, zorder=10)
    if fit_exp == True:
        # Fit the dataset to the exponential function
        params, cov = curve_fit(exponential_falloff, hist_central_x_val, hist, p0 = [max(hist), np.mean(hist_central_x_val), np.std(hist_central_x_val)])
        x_fit = np.linspace(0, endpoint, 500) 
        y_fit = exponential_falloff(x_fit, *params)
        plt.plot(x_fit, y_fit, label='Fitted exponential falloff', color='purple')
    plt.title('Radial distribution of OB association surface density')
    plt.xlabel('Heliocentric distance r (kpc)')
    plt.ylabel('$\\rho(r)$ (OB associations / kpc$^{-2}$)')
    plt.ylim(0, max(hist) * 1.5) # to limit the exponential curve from going too high
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()
    return bins, hist


def my_data_for_plotting():
    file_path = f'{const.FOLDER_OBSERVATIONAL_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    #print(data.describe()) # Get basic statistics
    age = data['Age(Myr)']
    distance = data['Distance (pc)'] / 1000
    glon = np.radians(data['l (deg)']) # convert to radians
    glat = np.radians(data['b (deg)']) # convert to radians
    rho = ut.rho(distance, glon, 0)
    theta = ut.theta(distance, glon, 0)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    z = ut.z(distance, glat)
    return x, y, z, distance, age


def plot_my_data(step=0.5): 
    """ Get my data and plot the distance histogram, age histogram and the associations in the galactic plane
    
    Args:
        step: float. Step size for the radial binning of associations in kpc
        
    Returns:
        None. Saves the plots
    """
    x, y, distance, age = my_data_for_plotting()
    plot_age_hist(age, filename='my_data_age_hist.pdf')
    plot_distance_hist(distance, filename='my_data_distance_hist.pdf', step=step, fit_exp=True)
    plot_associations(x, y, filename='my_data_associations.pdf', label_plotted_asc='Known associations', step=step)


def data_wright(filter_data=False, step=0.5):
    """ Get the Wright et al. (2020) data and plot the distance histogram and the associations
    
    Args:
        filter_data: bool. If True, filters out the data with 'Code' = 'C'
        step: float. Step size for the radial binning of associations in kpc
    
    Returns:
        None. Saves the plots
    """
    WRIGHT_CATALOGUE = "J/other/NewAR/90.1549"
    WRIGHT_TABLE = "table1"
    WRIGHT_COLUMNS = ['Name', 'Code', 'GLON', 'GLAT', 'Dist', 'Age']
    tap_records = obs_ut.get_catalogue_data(WRIGHT_CATALOGUE, WRIGHT_TABLE, WRIGHT_COLUMNS)
    print("Number of datapoints: ", len(tap_records))
    # filter out the datapoints with 'Code' = 'C'
    if filter_data == True:
        mask = tap_records['Code'] != 'C'
    else:
        mask = np.ones(len(tap_records), dtype=bool)
    wright_name = tap_records['Name'].data[mask]
    wright_code = tap_records['Code'].data[mask]
    wright_glon = np.radians(tap_records['GLON'].data[mask]) # convert to radians
    wright_glat = np.radians(tap_records['GLAT'].data[mask]) # convert to radians
    wright_distance = tap_records['Dist'].data[mask] / 1000
    wright_age = tap_records['Age'].data[mask]
    wright_rho = ut.rho(wright_distance, wright_glon, 0)
    wright_theta = ut.theta(wright_distance, wright_glon, 0)
    wright_x = wright_rho * np.cos(wright_theta) 
    wright_y = wright_rho * np.sin(wright_theta) 
    print("Number of datapoints after filtering: ", len(wright_name))
    plot_distance_hist(wright_distance, filename=f'wright_distance_hist_mask_{filter_data}.pdf', step=step, fit_exp=True)
    plot_associations(wright_x, wright_y, filename=f'wright_associations_arms_mask_{filter_data}.pdf', label_plotted_asc='Known associations',step=step)
    plot_age_hist(wright_age, filename=f'wright_age_mask_{filter_data}.pdf')
    

def modelled_data(galaxy):
    """ Get the modelled data and return the x, y, r and ages of the associations"""
    associations = galaxy.associations
    x_modelled = np.array([asc.x for asc in associations])
    y_modelled = np.array([asc.y for asc in associations])
    z_modelled = np.array([asc.z for asc in associations])
    r_modelled = np.sqrt(x_modelled ** 2 + (y_modelled - const.r_s) ** 2 + z_modelled**2) # subtract the distance from the Sun to the Galactic center in order to get the heliocentric distance
    ages_modelled = np.array([asc.age for asc in associations]) # at the moment (13.03.2023) the ages are equal to the creation time of the associations
    return x_modelled, y_modelled, r_modelled, ages_modelled


def plot_modelled_galaxy(galaxy, step=0.5, endpoint=5):
    """  Plot the modelled associations and the distance histogram"""
    x_modelled, y_modelled, r_modelled, ages_modelled = modelled_data(galaxy)
    plot_associations(x_modelled, y_modelled, filename='modelled_galaxy_associations.pdf', label_plotted_asc='Modelled associations', step=step)
    plot_distance_hist(r_modelled, 'modelled_galaxy_distance_hist.pdf', step, endpoint=endpoint, fit_exp=False)


def add_modelled_associations_to_observed(galaxy, step=0.5, endpoint=25):
    bins = np.arange(0, endpoint + step, step)
    known_associations = known_associations_to_association_class()
    distance_obs = np.array([asc.r for asc in known_associations])
    modelled_associations = galaxy.associations
    _, _, r_modelled, _ = modelled_data(galaxy)
    hist_modelled, _ = np.histogram(r_modelled, bins=bins)
    hist_obs, _ = np.histogram(distance_obs, bins=bins)
    associations_added = np.array([])
    for i in range(len(bins[1:])):
        diff = hist_modelled[i] - hist_obs[i] # difference between the number of modelled and observed associations in the bin
        mask_modelled = (r_modelled >= bins[i]) & (r_modelled < bins[i + 1]) # pick out the modelled associations which are in the bin
        if diff == hist_modelled[i]: # there are no observed associations in the bin
            associations_added = np.concatenate((associations_added, modelled_associations[mask_modelled])) # add all modelled associations in the bin
        elif diff > 0: # if there are more modelled associations in the bin than observed
            associations_added = np.concatenate((associations_added, rng.choice(modelled_associations[mask_modelled], size=diff))) # add diff associations randomly from the modelled associations in the bin
        elif diff < 0: # if there are more observed associations in the bin than modelled
            pass # do nothing
    return known_associations, associations_added


def plot_modelled_and_known_associations(modelled_galaxy, step=0.5, endpoint=25):
    known_associations, associations_added = add_modelled_associations_to_observed(modelled_galaxy, step, endpoint)
    x_obs = np.array([asc.x for asc in known_associations])
    y_obs = np.array([asc.y for asc in known_associations])
    x_added = np.array([asc.x for asc in associations_added])
    y_added = np.array([asc.y for asc in associations_added])
    # Now plot the modelled and known associations together
    fig, ax = plt.subplots(figsize=(10, 6))
    add_associations_to_ax(ax, x_obs, y_obs, 'Known associations', 'blue') # want the known associations to get its own label
    add_associations_to_ax(ax, x_added, y_added, 'Modelled associations', 'green')
    add_heliocentric_circles_to_ax(ax, step=step)
    add_spiral_arms_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', label='Galactic centre', s=15, zorder=11)
    plt.title('Distribution of known and modelled associations in the Galactic plane')
    plt.xlabel('x (kpc)')
    plt.ylabel('y (kpc)')
    plt.xlim(-7.5, 7.5)
    plt.ylim(-2, 12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True, zorder=-10)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/combined_associations.pdf')
    plt.close()


def calc_snps_known_association(n, min_mass, max_mass, association_age):
    # n = number of snps in the given mass range
    # min_mass = minimum mass for the mass range
    # function shall use the IMF to draw snps until we have n snps with mass >= min_mass, and then we will keep all snps with mass >= 8 solar masses
    # mass_range_snps = np.arange(8, 120 + 0.01, 0.01) # mass in solar masses. Used to draw random masses for the SNPs in the association
    m3 = np.arange(1.0, 120, 0.01)
    m3 = np.concatenate((m3, [120])) # add 120 solar masses to the array
    imf3 = ut.imf_3(m3)
    imf3 = imf3 / np.sum(imf3) # normalize
    n_drawn = 0
    n_matched = 0
    drawn_masses = []
    while n_matched < n:
        drawn_mass = rng.choice(m3, size=1, p=imf3)
        drawn_mass_age = ut.lifetime_as_func_of_initial_mass(drawn_mass)
        if drawn_mass >= 8: # if drawn mass is greater than or equal to 8 solar masses, keep it
            n_drawn += 1
            drawn_masses.append(drawn_mass)
        if drawn_mass >= min_mass and drawn_mass <= max_mass and drawn_mass_age >= association_age:
            # if drawn mass is within the given mass range and the age of the drawn mass is greater than or equal to the age of the association, keep it
            # this essentially means that if the drawn star could have survived up until today and match the mass criteria, increase the counter
            n_matched += 1
    return n_drawn, np.array(drawn_masses)


@ut.timing_decorator
def stat_one_asc(n, min_mass, max_mass, association_age, num_iterations=10000): 
    """ Calculate the statistics for one association
    
    Args:
        n: int. Number of stars in the association in the mass range
        min_mass: float. Minimum mass for the mass range
        max_mass: float. Maximum mass for the mass range
        association_age: float. Age of the association in Myr
        num_iterations: int. Number of iterations for the simulation
    
    Returns:
        n_drawn_mean: float. Mean number of drawn stars
        n_drawn_std: float. Standard deviation of the number of drawn stars
        exploded_sn_mean: float. Mean number of exploded supernovae
        exploded_sn_std: float. Standard deviation of the number of exploded supernovae
        exploded_sn_1_myr_mean: float. Mean number of exploded supernovae within 1 Myr
        exploded_sn_1_myr_std: float. Standard deviation of the number of exploded supernovae within 1 Myr
        stars_still_existing_mean: float. Mean number of stars still existing
        stars_still_existing_std: float. Standard deviation of the number of stars still existing
    """
    array_n_drawn = []
    array_exploded_sn = []
    array_exploded_sn_1_myr = []
    array_stars_still_existing = []
    for i in range(num_iterations):
        n_drawn, drawn_masses = calc_snps_known_association(n, min_mass, max_mass, association_age)
        drawn_ages = ut.lifetime_as_func_of_initial_mass(drawn_masses)
        array_n_drawn.append(n_drawn)
        mask_exploded = 0 <= association_age - drawn_ages # mask for the drawn stars which have exploded, i.e. the drawn stars which have a lifetime less than the age of the association
        mask_exploded_1_myr = association_age - drawn_ages <= 1  # mask for the drawn stars which have exploded within 1 Myr
        mask_exploded_1_myr = mask_exploded[mask_exploded_1_myr]
        mask_still_existing = association_age - drawn_ages < 0 # mask for the drawn stars which are still existing (lifetime of the star is greater than the age of the association)
        array_exploded_sn.append(np.sum(mask_exploded))
        array_exploded_sn_1_myr.append(np.sum(mask_exploded_1_myr))
        array_stars_still_existing.append(np.sum(mask_still_existing))
    n_drawn_mean = np.round(np.mean(array_n_drawn))
    n_drawn_std = np.round(np.std(array_n_drawn))
    exploded_sn_mean = np.round(np.mean(array_exploded_sn))
    exploded_sn_std = np.round(np.std(array_exploded_sn))
    exploded_sn_1_myr_mean = np.round(np.mean(array_exploded_sn_1_myr))
    exploded_sn_1_myr_std = np.round(np.std(array_exploded_sn_1_myr))
    stars_still_existing_mean = n_drawn_mean - exploded_sn_mean
    stars_still_existing_covariance = np.cov(array_n_drawn, array_exploded_sn)[0,1] # calculate the sample covariance. [0,1] is the covariance between the two arrays (np.cov returns a covariance matrix)
    # apply Bessel's correction for the variances
    combined_variance = np.var(array_n_drawn, ddof=1) + np.var(array_exploded_sn, ddof=1) - 2 * stars_still_existing_covariance
    combined_variance = np.max((0, combined_variance)) # if the combined variance is negative, set it to 0
    stars_still_existing_std = np.round(np.sqrt(combined_variance)) 
    return n_drawn_mean, n_drawn_std, exploded_sn_mean, exploded_sn_std, exploded_sn_1_myr_mean, exploded_sn_1_myr_std, stars_still_existing_mean, stars_still_existing_std


def my_data_for_simulation():
    """ Get the data on the know associations for the simulation
    
    Returns:
        association_name: array. Name of the association
        n: array. Number of stars in the association
        min_mass: array. Minimum mass for the mass range
        max_mass: array. Maximum mass for the mass range
        age: array. Age of the association in Myr
    """
    file_path = f'{const.FOLDER_OBSERVATIONAL_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    association_name = data['Name']
    n = data['Number of stars']
    min_mass = data['Min mass']
    max_mass = data['Max mass']
    age = data['Age(Myr)']
    return association_name, n, min_mass, max_mass, age


def stat_known_associations(num_iterations = 10):
    """ Calculate the statistics for the known associations and save the results to a CSV file"""
    
    association_name, n, min_mass, max_mass, age = my_data_for_simulation()
    # Prepare lists to store the statistics
    mean_snp_per_association = []
    std_snp_per_association = []
    mean_exploded_sn = []
    std_exploded_sn = []
    mean_exploded_sn_1_myr = []
    std_exploded_sn_1_myr = []
    mean_stars_still_existing = []
    std_stars_still_existing = []
    # Run the simulation and gather statistics for each association
    for i in range(len(association_name)):
        (n_drawn_mean, n_drawn_std, exploded_sn_mean, exploded_sn_std,
         exploded_sn_1_myr_mean, exploded_sn_1_myr_std,
         stars_still_existing_mean, stars_still_existing_std) = stat_one_asc(n[i], min_mass[i], max_mass[i], age[i], num_iterations)
        # Append the results to their respective lists
        mean_snp_per_association.append(n_drawn_mean)
        std_snp_per_association.append(n_drawn_std)
        mean_exploded_sn.append(exploded_sn_mean)
        std_exploded_sn.append(exploded_sn_std)
        mean_exploded_sn_1_myr.append(exploded_sn_1_myr_mean)
        std_exploded_sn_1_myr.append(exploded_sn_1_myr_std)
        mean_stars_still_existing.append(stars_still_existing_mean)
        std_stars_still_existing.append(stars_still_existing_std)
    # Create a DataFrame with the collected statistics
    df = pd.DataFrame({
        'Mean SNP born': mean_snp_per_association,
        'Std SNP born': std_snp_per_association,
        'Mean Exploded SN': mean_exploded_sn,
        'Std Exploded SN': std_exploded_sn,
        'Mean Exploded SN 1 Myr': mean_exploded_sn_1_myr,
        'Std Exploded SN 1 Myr': std_exploded_sn_1_myr,
        'Mean Stars Still Existing': mean_stars_still_existing,
        'Std Stars Still Existing': std_stars_still_existing
    }, index=association_name)
    # Save the DataFrame to a CSV file
    df.to_csv(f'{const.FOLDER_OBSERVATIONAL_DATA}/statistics_known_associations.csv')
    return


def known_associations_to_association_class():
    x, y, z, distance, age = my_data_for_plotting()
    association_name, n, min_mass, max_mass, age = my_data_for_simulation()
    associations = []
    for i in range(len(x)):
        num_snp, _ = calc_snps_known_association(n[i], min_mass[i], max_mass[i], age[i])
        # the association is created age[i] myrs ago with nump_snp snps, which are found to be the number needed to explain the observed number of stars in the association. 
        association = asc.Association(x[i], y[i], z[i], age[i], c=1, n=num_snp)
        # Next: update the snps to the present time
        association.update_sn(0)
        associations.append(association) # append association to list
    return associations


def combine_model_and_known_associations(galaxy, step=0.5, endpoint=25):
    known_associations, associations_added = add_modelled_associations_to_observed(galaxy, step, endpoint)
    pass



def main():
    step = 0.5
    """ plot_my_data(step=step)
    data_wright(True, step=step)
    data_wright(False, step=step) """
    """ galaxy = gal.Galaxy(10)
    plot_modelled_galaxy(galaxy, step=step, endpoint=25)
    plot_modelled_and_known_associations(galaxy, step=step, endpoint=25)  """
    stat_known_associations()



if __name__ == '__main__':
    main()


# When I am going to add associations from my model, I have two choices:
    # 1. Draw the associations from the model/ the same Galaxy which I am making the comparison with, until I have the same number of associations in each ring as in the model
    # 2. Draw new associations, not from the same Galaxy for which the comparison is made, but straight from the Association class itself. 