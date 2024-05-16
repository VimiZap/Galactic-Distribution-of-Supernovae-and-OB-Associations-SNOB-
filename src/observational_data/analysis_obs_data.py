import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import src.observational_data.obs_utilities as obs_ut
import src.utilities.utilities as ut
import src.nii_intensities.spiral_arm_model as sam
import src.utilities.constants as const
import src.galaxy_model.galaxy_class as gal
import src.galaxy_model.association_class as asc
from matplotlib.ticker import AutoMinorLocator
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
rng = np.random.default_rng()

SLOPE = 1_8 # slope for the power law distribution of the number of stars in the associations. Corresponds to alpha = 0.8


def modelled_data(galaxy):
    """ Get the modelled data and return the x, y, r and ages of the associations"""
    associations = galaxy.associations
    x_modelled = np.array([asc.x for asc in associations])
    y_modelled = np.array([asc.y for asc in associations])
    z_modelled = np.array([asc.z for asc in associations])
    r_modelled = np.sqrt(x_modelled ** 2 + (y_modelled - const.r_s) ** 2 + z_modelled**2) # subtract the distance from the Sun to the Galactic center in order to get the heliocentric distance
    ages_modelled = np.array([asc.age for asc in associations]) # at the moment (13.03.2023) the ages are equal to the creation time of the associations
    return x_modelled, y_modelled, r_modelled, ages_modelled


def my_data_for_stat():
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


def plot_age_hist(age_data_known, age_data_modelled, filename, bin_max_age: int = 50):
    """ Plot the age vs. distance of OB associations
    
    Args:
        age_data: array. Age of the associations
    
    Returns:
        None. Shows the plot
    """
    binwidth = 1
    bin_max_age = np.max(age_data_modelled)
    bins = np.arange(0, bin_max_age + binwidth, binwidth)
    plt.figure(figsize=(10, 6))
    plt.hist(age_data_known, bins=bins, label='Known associations', alpha=0.5)
    plt.hist(age_data_modelled, bins=bins, label='Modelled associations', alpha=0.5)
    #plt.title('Histogram of ages of OB associations')
    plt.xlabel('Age (Myr)', fontsize=12)
    plt.xlim(0, bin_max_age)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10)) 
    plt.ylabel('Counts', fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # Set the y-axis to only use integer ticks
    plt.grid(axis='y')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
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


def plot_distance_hist(heliocentric_distance_known, heliocentric_distance_modelled, filename, step=0.5, endpoint=2.5):
    """ Plot the histogram of distances of OB associations and fit the data to a Gaussian function

    Args:
        heliocentric_distance: array. Heliocentric distances of the associations in kpc
        filename: str. Name of the file to save the plot
        step: float. Step size for the histogram in kpc
        endpoint: float. Max radial distance in kpc
    
    Returns:
        None. Saves the plot
    """
    bins = np.arange(0, endpoint + step, step)
    area_per_circle = area_per_bin(bins)
    hist_known, _ = np.histogram(heliocentric_distance_known, bins=bins)
    hist_modelled, _ = np.histogram(heliocentric_distance_modelled, bins=bins)
    hist_known = hist_known / area_per_circle # find the surface density of OB associations
    hist_modelled = hist_modelled / area_per_circle
    hist_central_x_val = bins[:-1] + step / 2 # central x values for each bin
    # Make the histogram    
    plt.figure(figsize=(10, 6))
    plt.bar(hist_central_x_val, hist_known, width=step, alpha=0.5, label='Known associations')
    plt.bar(hist_central_x_val, hist_modelled, width=step, alpha=0.5, label='Modelled associations')
    #plt.title('Radial distribution of OB association surface density')
    plt.xlabel('Heliocentric distance r (kpc)', fontsize=12)
    plt.xlim(0, endpoint)
    plt.ylabel('$\\rho(r)$ (OB associations / kpc$^{-2}$)', fontsize=12)
    plt.grid(axis='y')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()
    return bins, hist_known


def add_heliocentric_circles_to_ax(ax, step=0.5, linewidth=1):
    """ Add heliocentric circles to the plot
    
    Args:
        ax: axis. The axis to add the circles to
        step: float. Step size for the radial binning of associations in kpc
    
    Returns:
        None
    """
    thetas_heliocentric_circles = np.linspace(0, 2 * np.pi, 100)
    for i in range(1, 7):
        x_heliocentric_circles = i * step * np.cos(thetas_heliocentric_circles)
        y_heliocentric_circles = i * step * np.sin(thetas_heliocentric_circles) + const.r_s
        ax.plot(x_heliocentric_circles, y_heliocentric_circles, color='black', linestyle='--', linewidth=linewidth, zorder=5) # plot the heliocentric circles
    return


def add_spiral_arms_to_ax(ax, linewidth=3):
    """ Add the spiral arms to the plot
    
    Args:
        ax: axis. The axis to add the spiral arms to
    
    Returns:
        None
    """
    colors = sns.color_palette('bright', 7)
    rho_min_array = const.rho_min_spiral_arm
    rho_max_array = const.rho_max_spiral_arm
    for i in range(len(const.arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(const.arm_angles[i], const.pitch_angles[i], rho_min=rho_min_array[i], rho_max=rho_max_array[i])
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        ax.plot(x, y, linewidth = linewidth, zorder=6, markeredgewidth=1, markersize=1, color=colors[i]) # plot the spiral arm medians
    return


def add_associations_to_ax(ax, x, y, label, color, s=15):
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
    ax.scatter(x, y, color=color, alpha=0.5, s=s, label=label, zorder=10)
    return


def add_spiral_arm_names_to_ax(ax, fontsize=20):
    """ Add the names of the spiral arms to the plot
    
    Args:
        ax: axis. The axis to add the spiral arm names to
    
    Returns:
        None
    """
    text_x_pos = [-3.5, -5, -6.2, -6.8]
    text_y_pos = [2.8, 4.9, 6.7, 10.1]
    text_rotation = [24, 23, 20, 16]
    text_arm_names = ['Norma-Cygnus', 'Scutum-Crux', 'Sagittarius-Carina', 'Perseus']
    
    for i in range(len(const.arm_angles[:-1])): # skip local arm
        ax.text(text_x_pos[i], text_y_pos[i], text_arm_names[i], fontsize=fontsize, zorder=20, rotation=text_rotation[i],
                weight='bold', bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))
    return


def add_local_arm_to_ax(ax):
    """ Add the local arm to the plot
    
    Args:
        ax: axis. The axis to add the spiral arms to
    
    Returns:
        None
    """
    arm_index=4
    rho_min_local = const.rho_min_spiral_arm[arm_index]
    rho_max_local = const.rho_max_spiral_arm[arm_index]
    theta, rho = sam.spiral_arm_medians(arm_angle=const.arm_angles[arm_index], pitch_angle=const.pitch_angles[arm_index], rho_min=rho_min_local, rho_max=rho_max_local)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    ax.plot(x, y, color='black', marker='o', linewidth = 0.0001, zorder=6, markeredgewidth=0.0001, markersize=0.0001)
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
    add_spiral_arm_names_to_ax(ax)
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


def plot_age_vs_distance(age, distance, filename):
    """ Plot the age vs. distance of OB associations
    
    Args:
        age: array. Age of the associations
        distance: array. Distance of the associations
    
    Returns:
        None. Saves the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(distance, age, color='green', s=8, alpha=0.5, zorder=10)
    plt.title('Age vs. distance of OB associations')
    plt.ylabel('Age (Myr)')
    plt.xlabel('Distance (kpc)')
    plt.grid(True, zorder=0)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()


def plot_my_data(step=0.5): 
    """ Get my data and plot the distance histogram, age histogram and the associations in the galactic plane
    
    Args:
        step: float. Step size for the radial binning of associations in kpc
        
    Returns:
        None. Saves the plots
    """
    x, y, z, distance, age = my_data_for_plotting()
    plot_associations(x, y, filename='my_data_associations.pdf', label_plotted_asc='Known associations', step=step)
    plot_distance_hist(distance, filename='my_data_distance_hist.pdf', step=step)
    plot_age_hist(age, filename='my_data_age_hist.pdf')
    plot_age_vs_distance(age, distance, filename='my_data_age_vs_distance.pdf')


def plot_data_wright(filter_data=False, step=0.5):
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
    plot_associations(wright_x, wright_y, filename=f'wright_associations_arms_mask_{filter_data}.pdf', label_plotted_asc='Known associations',step=step)
    plot_distance_hist(wright_distance, filename=f'wright_distance_hist_mask_{filter_data}.pdf', step=step)
    plot_age_hist(wright_age, filename=f'wright_age_mask_{filter_data}.pdf')
    plot_age_vs_distance(wright_age, wright_distance, filename=f'wright_age_vs_distance_mask_{filter_data}.pdf')
    

def plot_modelled_galaxy(galaxy, step=0.5, endpoint=5):
    """  Plot the modelled associations, the distance histogram and the age histogram"""
    x_modelled, y_modelled, r_modelled, ages_modelled = modelled_data(galaxy)
    plot_associations(x_modelled, y_modelled, filename='modelled_galaxy_associations.pdf', label_plotted_asc='Modelled associations', step=step)
    plot_distance_hist(r_modelled, 'modelled_galaxy_distance_hist.pdf', step, endpoint=endpoint)


def calc_snps_known_association(n, min_mass, max_mass, association_age):
    """ Calculate the number of drawn stars and their masses for a known association. Takes into account the mass range of the observed stars today and the age of the association.
    The returned number of stars is an estimate on how many stars had to form in the association an 'association_age' years ago to have 'n' stars today.
    
    Args:
        n: int. Number of stars in the association in the given mass range
        min_mass: float. Minimum mass for the mass range
        max_mass: float. Maximum mass for the mass range
        association_age: float. Age of the association in Myr
    
    Returns:
        n_drawn: int. Number of drawn stars
        drawn_masses: array. Masses of the drawn stars
    """
    m3 = np.arange(1.0, 120, 0.01) # mass in solar masses. Used to draw random masses for the SNPs in the association
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


def calc_num_snps_known_associations_batch():
    """ Calculate the number of drawn stars for the known associations. Uses calc_snps_known_association() to calculate the number of drawn stars for each association, but the masses are discarded.
    
    Returns:
        n_drawn_list: array. Number of drawn stars for each association
    """
    association_name, n, min_mass, max_mass, age = my_data_for_stat()
    n_drawn_list = []
    for i in range(len(association_name)):
        n_drawn, _ = calc_snps_known_association(n[i], min_mass[i], max_mass[i], age[i])
        n_drawn_list.append(n_drawn)
    return np.array(n_drawn_list)


@ut.timing_decorator
def stat_one_known_asc(n, min_mass, max_mass, association_age, num_iterations=10000): 
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


@ut.timing_decorator
def stat_known_associations(num_iterations = 10):
    """ Calculate the statistics for the known associations and save the results to a CSV file.
    Calculates the mean and standard deviation of the number of drawn stars, the number of exploded supernovae, the number of exploded supernovae within 1 Myr and the number of stars still existing for each association."""
    association_name, n, min_mass, max_mass, age = my_data_for_stat()
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
         stars_still_existing_mean, stars_still_existing_std) = stat_one_known_asc(n[i], min_mass[i], max_mass[i], age[i], num_iterations)
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
    """ Convert the known associations to the Association class
    
    Returns:
        associations: list. List of Association objects
    """
    x, y, z, distance, age = my_data_for_plotting()
    num_snp = calc_num_snps_known_associations_batch()
    #############################################################print(num_snp)
    associations = []
    for i in range(len(x)):
        # the association is created age[i] myrs ago with nump_snp[i] snps, which are found to be the number needed to explain the observed number of stars in the association. 
        association = asc.Association(x[i], y[i], z[i], age[i], c=1, n=[num_snp[i]])
        # Next: update the snps to the present time
        association.update_sn(0)
        associations.append(association) # append association to list
    return associations


def combine_modelled_and_known_associations(modelled_associations, step=0.5, endpoint=25):
    """ Combine the modelled and known associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
        step: float. Step size for the radial binning of associations in kpc
        endpoint: float. Max radial distance in kpc
    
    Returns:
        known_associations: array. Known associations
        associations_added: array. Modelled associations added to the known associations
    """
    bins = np.arange(0, endpoint + step, step)
    known_associations = known_associations_to_association_class()
    distance_obs = np.array([asc.r for asc in known_associations])
    #modelled_associations = modelled_galaxy.associations
    #_, _, r_modelled, _ = modelled_data(modelled_galaxy) # retrieve the radial distances of the modelled associations
    r_modelled = np.array([np.sqrt(asc.x**2 + (asc.y - const.r_s)**2 + asc.z**2) for asc in modelled_associations])
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
    """ Plot the modelled and known associations together
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
        step: float. Step size for the radial binning of associations in kpc
        endpoint: float. Max radial distance in kpc
    
    Returns:
        None. Saves the plot
    """
    modelled_associations = modelled_galaxy.associations
    print(f'Number of associations added: {len(modelled_associations)}')
    asc_modelled_masses = np.array([np.sum(asc.star_masses) for asc in modelled_associations])
    modelled_associations = modelled_associations[asc_modelled_masses > 7] # remove associations with no stars
    known_associations, associations_added = combine_modelled_and_known_associations(modelled_associations, step, endpoint)
    asc_mass_added = np.array([np.sum(asc.star_masses) for asc in associations_added])
    associations_added = associations_added[asc_mass_added > 7] # remove associations with no stars 
    print(f'Number of associations added with stars: {len(associations_added)}')
    x_obs = np.array([asc.x for asc in known_associations])
    y_obs = np.array([asc.y for asc in known_associations])
    x_added = np.array([asc.x for asc in associations_added])
    y_added = np.array([asc.y for asc in associations_added])
    # Now plot the modelled and known associations together
    fig, ax = plt.subplots(figsize=(20, 18))
    add_associations_to_ax(ax, x_obs, y_obs, 'Known associations', 'blue', s=40) # want the known associations to get its own label
    add_associations_to_ax(ax, x_added, y_added, 'Modelled associations', 'darkgreen', s=40)
    add_heliocentric_circles_to_ax(ax, step=step, linewidth=1)
    add_spiral_arms_to_ax(ax, linewidth=3)
    add_spiral_arm_names_to_ax(ax, fontsize=25)
    #add_local_arm_to_ax(ax)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=45, zorder=11)
    ax.scatter(0, 0, color='black', marker='o', s=50, zorder=11)
    ax.text(-0.38, 0.5, 'GC', fontsize=35, zorder=7)
    #plt.title(f'Distribution of known and modelled associations in the Galactic plane. \n Galaxy generated {sim_time} Myrs ago')
    plt.xlabel('$x$ (kpc)', fontsize=35)
    plt.ylabel('$y$ (kpc)', fontsize=35)
    plt.xlim(-7.5, 7.5)
    plt.ylim(-2, 12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tick_params(axis='both', which='major', labelsize=35)
    legend = plt.legend(framealpha=0.9, fontsize=30, loc='upper right')
    legend.set_zorder(50)
    plt.grid(True, zorder=-10)
    plt.rc('font', size=50) # increase the font size
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/combined_associations_{SLOPE}.pdf')
    plt.close()


def calc_avg_mass_hist(num_iterations: int = 10):
    mass_step = 8
    bins = np.arange(8, 120 + mass_step, mass_step)
    print(bins)
    hist_known = np.zeros((num_iterations, len(bins) - 1))
    hist_added = np.zeros((num_iterations, len(bins) - 1))
    for it in range(num_iterations):
        modelled_galaxy = gal.Galaxy(10, read_data_from_file=True)
        known_associations, associations_added = combine_modelled_and_known_associations(modelled_galaxy)
        known_masses = np.array([])
        added_masses = np.array([])
        for asc in known_associations:
            known_masses = np.concatenate((known_masses, asc.star_masses))
        for asc in associations_added:
            added_masses = np.concatenate((added_masses, asc.star_masses))
        hist_known_it, _ = np.histogram(known_masses, bins=bins)
        hist_added_it, _ = np.histogram(added_masses, bins=bins)
        hist_known[it] = hist_known_it
        hist_added[it] = hist_added_it
    hist_known_mean = np.mean(hist_known, axis=0)
    hist_added_mean = np.mean(hist_added, axis=0)
    return bins, hist_known_mean, hist_added_mean     


def plot_mass_hist():
    """ Plot the histogram of star masses for known and modelled associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
    
    Returns:
        None. Saves the plot
    """

    """ 
    known_associations, associations_added = combine_modelled_and_known_associations(modelled_galaxy)
    known_masses, added_masses = np.array([]), np.array([])
    for asc in known_associations:
        known_masses = np.concatenate((known_masses, asc.star_masses))
    for asc in associations_added:
        added_masses = np.concatenate((added_masses, asc.star_masses))    
    bins = np.arange(8, 120 + 8, 8) """

    bins, hist_known_mean, hist_added_mean = calc_avg_mass_hist()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)
    #ax1.bar(bin_centers, hist, width=bin_widths, align='center')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(bin_centers, hist_known_mean, width=bin_widths, label='Known Associations')
    ax2.bar(bin_centers, hist_added_mean, width=bin_widths, label='Modelled Associations')
    ax1.set_xlabel('Mass (M$_\odot$)')
    ax1.set_ylabel('Number of Stars')
    ax2.set_xlabel('Mass (M$_\odot$)')
    ax2.set_ylabel('Number of Stars')
    ax1.set_title('Histogram of Star Masses (Known Associations)')
    ax2.set_title('Histogram of Star Masses (Modelled Associations)')
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/star_mass_hist.pdf')
    plt.close()


def calc_avg_num_stars_hist(num_iterations: int = 10):
    num_stars_step = 100
    bins = np.arange(0, 3000 + num_stars_step, num_stars_step)
    hist_known = np.zeros((num_iterations, len(bins) - 1))
    hist_added = np.zeros((num_iterations, len(bins) - 1))
    for it in range(num_iterations):
        modelled_galaxy = gal.Galaxy(10, read_data_from_file=True)
        known_associations, associations_added = combine_modelled_and_known_associations(modelled_galaxy)
        num_stars_known = np.array([asc.number_sn for asc in known_associations])
        num_stars_added = np.array([asc.number_sn for asc in associations_added])
        hist_known_it, _ = np.histogram(num_stars_known, bins=bins)
        hist_added_it, _ = np.histogram(num_stars_added, bins=bins)
        hist_known[it] = hist_known_it
        hist_added[it] = hist_added_it
    hist_known_mean = np.mean(hist_known, axis=0)
    hist_added_mean = np.mean(hist_added, axis=0)
    return bins, hist_known_mean, hist_added_mean


def plot_num_stars_hist():
    """ Plot the histogram of the number of stars per association for known and modelled associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
    
    Returns:
        None. Saves the plot
    """
    bins, hist_known_mean, hist_added_mean = calc_avg_num_stars_hist()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(bin_centers, hist_known_mean, width=bin_widths, label='Known Associations')
    ax2.bar(bin_centers, hist_added_mean, width=bin_widths, label='Modelled Associations')
    ax1.set_xlabel('Number of Stars')
    ax1.set_ylabel('Frequency')
    ax2.set_xlabel('Number of Stars')
    ax2.set_ylabel('Frequency')
    ax1.set_title('Histogram of Number of Stars per Association (Known Associations)')
    ax2.set_title('Histogram of Number of Stars per Association (Modelled Associations)')
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/num_stars_hist.pdf')
    plt.close()


def calc_avg_asc_mass_hist(modelled_galaxy, num_iterations: int = 10, bin_max_mass: int = 3000):
    asc_mass_step = 50
    bins = np.arange(0, bin_max_mass + asc_mass_step, asc_mass_step)
    modelled_associations = modelled_galaxy.associations
    mass_asc_modelled = np.array([np.sum(asc.star_masses) for asc in modelled_associations])
    mass_asc_modelled = mass_asc_modelled[mass_asc_modelled > 7] # remove associations with mass less than 8 solar masses (these appear due to SNP masses are set to zero once they die. Set > 7 to avoid rounding errors in the mass calculation)
    hist_modelled, _ = np.histogram(mass_asc_modelled, bins=bins)
    hist_known = np.zeros((num_iterations, len(bins) - 1))
    for it in range(num_iterations):
        if it % 10 == 0:
            logging.info(f'Iteration {it}')
        known_associations = known_associations_to_association_class()
        mass_asc_known = np.array([np.sum(asc.star_masses) for asc in known_associations])
        hist_known_it, _ = np.histogram(mass_asc_known, bins=bins)
        hist_known[it] = hist_known_it
    hist_known_mean = np.mean(hist_known, axis=0)
    return bins, hist_known_mean, hist_modelled


def plot_avg_asc_mass_hist(modelled_galaxy, num_iterations: int, star_formation_episodes: int, sim_time: int):
    """ Plot the histogram of the number of stars per association for known and modelled associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
    
    Returns:
        None. Saves the plot
    """
    logging.info('Plotting average association mass histogram')
    bin_max_mass = 2000
    bins, hist_known_mean, hist_added_mean = calc_avg_asc_mass_hist(modelled_galaxy, num_iterations=num_iterations, bin_max_mass=bin_max_mass)
    hist_added_mean = hist_added_mean / np.sum(hist_added_mean) 
    hist_known_mean = hist_known_mean / np.sum(hist_known_mean)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist_known_mean, width=bin_widths, label='Known Associations', alpha=0.5)
    plt.bar(bin_centers, hist_added_mean, width=bin_widths, label='Modelled Associations', alpha=0.5)
    plt.xlabel('Association mass (M$_\odot$)', fontsize=12)
    plt.xlim(0, bin_max_mass)
    plt.ylabel('Frequency', fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # Set the y-axis to only use integer ticks
    #plt.title('Histogram of association masses shown for modelled and known associations.')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10)) 
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/asc_mass_hist_{star_formation_episodes}_num_iterations_{num_iterations}_sim_time_{sim_time}_{SLOPE}.pdf')
    plt.close()


def calc_avg_asc_age_hist(num_iterations: int = 10):
    asc_age_step = 1
    bins = np.arange(0, 50 + asc_age_step, asc_age_step)
    hist_known = np.zeros((num_iterations, len(bins) - 1))
    hist_added = np.zeros((num_iterations, len(bins) - 1))
    for it in range(num_iterations):
        modelled_galaxy = gal.Galaxy(10, read_data_from_file=True)
        known_associations, associations_added = combine_modelled_and_known_associations(modelled_galaxy)
        age_asc_known = np.array([asc.age for asc in known_associations])
        age_asc_added = np.array([asc.age for asc in associations_added])
        hist_known_it, _ = np.histogram(age_asc_known, bins=bins)
        hist_added_it, _ = np.histogram(age_asc_added, bins=bins)
        hist_known[it] = hist_known_it
        hist_added[it] = hist_added_it
    hist_known_mean = np.mean(hist_known, axis=0)
    hist_added_mean = np.mean(hist_added, axis=0)
    return bins, hist_known_mean, hist_added_mean


def plot_avg_asc_age_hist():
    """ Plot the histogram of the number of stars per association for known and modelled associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
    
    Returns:
        None. Saves the plot
    """
    logging.info('Plotting average association age histogram')
    known_associations = known_associations_to_association_class()
    asc_age_step = 1
    bins = np.arange(0, 50 + asc_age_step, asc_age_step)
    age_asc_known = np.array([asc.age for asc in known_associations])
    hist_known, _ = np.histogram(age_asc_known, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist_known, width=bin_widths, label='Known Associations')
    plt.xlabel('Association age (Myr)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    #plt.title('Histogram of known association ages')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # Set the y-axis to only use integer ticks
    # add tick marks for x-axis
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10)) 
    plt.xlim(0, 50)
    # Add horizontal gridlines
    plt.grid(axis='y')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/asc_age_hist_{SLOPE}.pdf')
    plt.close()


def plot_distance_hist_known_modelled(modelled_galaxy, endpoint=2.5):
    """ Plot the histogram of the radial distances of known and modelled associations
    
    Args:
        modelled_galaxy: Galaxy. The modelled galaxy
    
    Returns:
        None. Saves the plot
    """
    logging.info('Plotting distance histogram of known and modelled associations')
    modelled_associations = modelled_galaxy.associations
    known_associations = known_associations_to_association_class()
    distance_known = np.array([asc.r for asc in known_associations])
    distance_modelled = np.array([asc.r for asc in modelled_associations])
    masses_asc_modelled = np.array([np.sum(asc.star_masses) for asc in modelled_associations])
    modelled_asc_mask = np.array([masses > 7 for masses in masses_asc_modelled]) # mask for the modelled associations with mass greater than 7 solar masses. > 7 to avoid rounding errors in the mass calculation, and this mask is to remove associations for which all SNPs have exploded
    distance_modelled = distance_modelled[modelled_asc_mask] # remove modelled associations which have no stars anymore
    print('------------------------------------- Number of associations: ', len(distance_modelled))
    plot_distance_hist(heliocentric_distance_known=distance_known, heliocentric_distance_modelled=distance_modelled, filename=f'histogram_dist_known_modelled_asc_{SLOPE}.pdf', endpoint=endpoint)


def plot_age_hist_known_modelled(modelled_galaxy):
    """ Plot the histogram of the ages of known and modelled associations within 2.5 kpc

    Returns:
        None. Saves the plot
    """
    logging.info('Plotting age histogram of known and modelled associations')
    modelled_associations = modelled_galaxy.associations
    known_associations = known_associations_to_association_class()
    age_known = np.array([asc.age for asc in known_associations])
    age_modelled = np.array([asc.age for asc in modelled_associations])
    masses_asc_modelled = np.array([np.sum(asc.star_masses) for asc in modelled_associations])
    modelled_asc_exist_mask = np.array([masses > 7 for masses in masses_asc_modelled]) # mask for the modelled associations with mass greater than 7 solar masses. > 7 to avoid rounding errors in the mass calculation, and this mask is to remove associations for which all SNPs have exploded
    asc_modelled_radial_distances = np.array([np.sqrt(asc.x**2 + (asc.y - const.r_s)**2 + asc.z**2) for asc in modelled_associations])
    modelled_asc_distance_mask = asc_modelled_radial_distances <= 2.5 # mask for the modelled associations which are within 2.5 kpc
    modelled_asc_mask_combined = modelled_asc_exist_mask & modelled_asc_distance_mask
    age_modelled = age_modelled[modelled_asc_mask_combined] # remove modelled associations which have no stars anymore
    #age_modelled = age_modelled[asc_modelled_radial_distances < 3] # remove modelled associations which are further away than 3 kpc
    print('------------------------------------- Number of associations: ', len(age_modelled))
    plot_age_hist(age_known, age_modelled, filename=f'histogram_age_known_modelled_asc_{SLOPE}.pdf')

 
def main():
    step = 0.5
    #plot_my_data(step=step)
    #plot_data_wright(True, step=step)
    #plot_data_wright(False, step=step)
    sim_time=100
    #plot_modelled_galaxy(galaxy, step=step, endpoint=25)
    galaxy_1 = gal.Galaxy(sim_time, read_data_from_file=True, star_formation_episodes=1)
    galaxy_3 = gal.Galaxy(sim_time, read_data_from_file=True, star_formation_episodes=3)
    galaxy_5 = gal.Galaxy(sim_time, read_data_from_file=True, star_formation_episodes=5)
    plot_modelled_and_known_associations(galaxy_5, step=step, endpoint=25) 
    #plot_mass_hist()
    #plot_num_stars_hist()
    #stat_known_associations(num_iterations=10000)
    num_iterations = 50
    plot_avg_asc_mass_hist(galaxy_1, num_iterations=num_iterations, star_formation_episodes=1, sim_time=sim_time)
    plot_avg_asc_mass_hist(galaxy_3, num_iterations=num_iterations, star_formation_episodes=3, sim_time=sim_time)
    plot_avg_asc_mass_hist(galaxy_5, num_iterations=num_iterations, star_formation_episodes=5, sim_time=sim_time)
    #plot_avg_asc_age_hist()
    plot_distance_hist_known_modelled(galaxy_5)
    plot_age_hist_known_modelled(galaxy_5)


if __name__ == '__main__':
    main()
