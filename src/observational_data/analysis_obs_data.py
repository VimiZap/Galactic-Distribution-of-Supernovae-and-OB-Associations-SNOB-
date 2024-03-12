import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import obs_utilities as obs_ut
from scipy.optimize import curve_fit
import src.utilities.utilities as ut
import src.spiral_arm_model as sam

import logging

r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center
arm_angles = np.radians([65, 160, 240, 330])  # best fit for the new r_s
pitch_angles = np.radians([14, 14, 14, 16]) # best fir to new r_s

FOLDER_OUTPUT = 'data/plots/observational_plots'
FOLDER_OBS_DATA = 'data/observational'


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
    plt.savefig(f'{FOLDER_OUTPUT}/{filename}')
    plt.close()


def plot_associations(glon, heliocentric_distance, filename, step=500):
    """ Plot the distribution of known associations in the Galactic plane together with the spiral arms medians
    
    Args:
        glon: array. Galactic longitude of associations to be plotted
        heliocentric_distance: array. Heliocentric distance of associations to be plotted
        filename: str. Name of the file to save the plot
        step: int. Step size for the radial binning of associations in pc
    
    Returns:
        None. Saves the plot
    """
    rho = ut.rho(heliocentric_distance, glon, 0)
    theta = ut.rho(heliocentric_distance, glon, 0)
    x = rho * np.cos(theta) / 1000 # convert to kpc
    y = rho * np.sin(theta) / 1000 + r_s # convert to kpc, and add the distance from the Sun to the Galactic center
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.5, s=8, zorder=10, label='Known associations')
    plt.scatter(0, r_s, color='red', marker='o', label='Sun', s=10)
    #plt.scatter(0, 0, color='black', marker='o', label='Galactic center')
    for i in range(len(arm_angles)):
        # generate the spiral arm medians
        theta, rho = sam.spiral_arm_medians(arm_angles[i], pitch_angles[i])
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        plt.plot(x, y, color='black', marker='o', linewidth = 0.0001, zorder=0, markeredgewidth=0.0001, markersize=0.0001) # plot the spiral arm medians
    thetas_heliocentric_circles = np.linspace(0, 2 * np.pi, 100)
    for i in range(1, 10):
        x_heliocentric_circles = i * step / 1000 * np.cos(thetas_heliocentric_circles)
        y_heliocentric_circles = i * step / 1000 * np.sin(thetas_heliocentric_circles) + r_s
        plt.plot(x_heliocentric_circles, y_heliocentric_circles, color='black', linestyle='--', linewidth=0.5, zorder=0) # plot the heliocentric circles
    plt.title('Distribution of known associations in the Galactic plane')
    plt.xlabel('x (kpc)')
    plt.ylabel('y (kpc)')
    plt.xlim(-7.5, 7.5)
    plt.ylim(2, 12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{FOLDER_OUTPUT}/{filename}')
    plt.close()


def area_per_bin(bins):
    """ Calculate the area of each bin in a histogram
    
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


def plot_distance_hist(heliocentric_distance, filename, step=500, endpoint=5000):
    """ Plot the histogram of distances of OB associations and fit the data to a Gaussian function

    Args:
        heliocentric_distance: array. Heliocentric distances of the associations
        filename: str. Name of the file to save the plot
        step: int. Step size for the histogram in pc
        endpoint: int. Max radial distance in pc
    
    Returns:
        None. Saves the plot
    """
    bins = np.arange(0, endpoint + step, step)
    area_per_circle = area_per_bin(bins)
    hist, _ = np.histogram(heliocentric_distance, bins=bins)
    hist = hist / area_per_circle # find the surface density of OB associations
    hist_central_x_val = bins[:-1] + step / 2 # central x values for each bin
    bins, hist_central_x_val = bins / 1000, hist_central_x_val / 1000 # convert to kpc
    avg_hist = hist[hist_central_x_val <= 2.5] # pick out the associations for r < 2.5 kpc
    avg_hist = np.mean(avg_hist) # average the associatins for r < 2.5 kpc
    # Make the histogram    
    plt.figure(figsize=(10, 6))
    plt.bar(hist_central_x_val, hist, width=step / 1000, color='green', alpha=0.7, zorder=10)
    # Fit the dataset to the Gaussian function
    params, cov = curve_fit(exponential_falloff, hist_central_x_val, hist, p0 = [max(hist), np.mean(hist_central_x_val), np.std(hist_central_x_val)])
    x_fit = np.linspace(0, endpoint, 500) / 1000 # convert to kpc
    y_fit = exponential_falloff(x_fit, *params)
    plt.plot(x_fit, y_fit, label='Fitted exponential falloff', color='purple')
    plt.title('Radial distribution of OB association surface density')
    plt.xlabel('Heliocentric distance r (kpc)')
    plt.ylabel('$\\rho(r)$ (OB associations / pc$^{-2}$)')
    plt.ylim(0, max(hist) * 1.5) # to limit the exponential curve from going too high
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(f'{FOLDER_OUTPUT}/{filename}')
    plt.close()


def data_wright(filter_data=False, step=500):
    """ Get the Wright et al. (2020) data and plot the distance histogram and the associations
    
    Args:
        filter_data: bool. If True, filters out the data with 'Code' = 'C'
    
    Returns:
        None
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
    wright_glon = tap_records['GLON'].data[mask]
    wright_glat = tap_records['GLAT'].data[mask]
    wright_distance = tap_records['Dist'].data[mask]
    wright_age = tap_records['Age'].data[mask]
    print("Number of datapoints after filtering: ", len(wright_name))
    plot_distance_hist(wright_distance, filename=f'wright_distance_hist_mask_{filter_data}.pdf', step=step)
    plot_associations(wright_glon, wright_distance, filename=f'wright_associations_arms_mask_{filter_data}.pdf', step=step)
    plot_age_hist(wright_age, filename=f'wright_age_mask_{filter_data}.pdf')


def main():
    file_path = f'{FOLDER_OBS_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    #print(data.describe()) # Get basic statistics
    step = 500 # pc, stepsize for the radial binning of associations
    plot_age_hist(data['Age(Myr)'], filename='my_data_age_hist.pdf')
    plot_distance_hist(data['Distance (pc)'], filename='my_data_distance_hist.pdf', step=step)
    plot_associations(data['l (deg)'], data['Distance (pc)'], filename='my_data_associations.pdf', step=step)
    data_wright(True, step=step)
    data_wright(False, step=step)


if __name__ == '__main__':
    main()