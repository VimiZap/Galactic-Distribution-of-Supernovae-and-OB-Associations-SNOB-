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
import logging
logging.basicConfig(level=logging.INFO)


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


def add_helicentric_circles_to_ax(ax, step=0.5):
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
        ax.plot(x_heliocentric_circles, y_heliocentric_circles, color='black', linestyle='--', linewidth=0.5, zorder=0) # plot the heliocentric circles
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
        ax.plot(x, y, color='black', marker='o', linewidth = 0.0001, zorder=0, markeredgewidth=0.0001, markersize=0.0001) # plot the spiral arm medians
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
    ax.scatter(x, y, color='blue', alpha=0.5, s=8, zorder=10, label=label_plotted_asc)
    ax.scatter(0, const.r_s, color='red', marker='o', label='Sun', s=10)
    add_helicentric_circles_to_ax(ax, step=step)
    add_spiral_arms_to_ax(ax)
    plt.title('Distribution of known associations in the Galactic plane')
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
    plt.ylabel('$\\rho(r)$ (OB associations / pc$^{-2}$)')
    plt.ylim(0, max(hist) * 1.5) # to limit the exponential curve from going too high
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(f'{const.FOLDER_OBSERVATIONAL_PLOTS}/{filename}')
    plt.close()


def data_mine(step=0.5): 
    """ Get my data and plot the distance histogram, age histogram and the associations in the galactic plane
    
    Args:
        step: float. Step size for the radial binning of associations in kpc
        
    Returns:
        None. Saves the plots
    """
    file_path = f'{const.FOLDER_OBSERVATIONAL_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    #print(data.describe()) # Get basic statistics
    age = data['Age(Myr)']
    distance = data['Distance (pc)'] / 1000
    glon = np.radians(data['l (deg)']) # convert to radians
    rho = ut.rho(distance, glon, 0)
    theta = ut.theta(distance, glon, 0)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
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
    

def modelled_galaxy(galaxy, step=0.5, endpoint=5):
    """  
    
    """
    xs = np.array([asc.x for asc in galaxy._galaxy])
    ys = np.array([asc.y for asc in galaxy._galaxy])
    rs = np.sqrt(xs ** 2 + (ys - const.r_s) ** 2) # subtract the distance from the Sun to the Galactic center in order to get the heliocentric distance
    plot_associations(xs, ys, filename='modelled_galaxy_associations.pdf', label_plotted_asc='Modelled associations', step=step)
    plot_distance_hist(rs, 'modelled_galaxy_distance_hist.pdf', step, endpoint=endpoint)


def main():
    step = 0.5
    data_mine(step=step)
    data_wright(True, step=step)
    data_wright(False, step=step)
    galaxy = gal.Galaxy(100)
    modelled_galaxy(galaxy, step=0.5, endpoint=25)


if __name__ == '__main__':
    main()
    