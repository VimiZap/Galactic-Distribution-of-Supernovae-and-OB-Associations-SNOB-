import pandas as pd
import matplotlib.pyplot as plt
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


def plot_age_distance(distance_data, age_data, filename):
    """ Plot the age vs. distance of OB associations
    
    Args:
        distance_data: array. Heliocentric distance of the associations
        age_data: array. Age of the associations
    
    Returns:
        None. Shows the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(distance_data, age_data, color='blue', alpha=0.5)
    plt.title('Distance vs. Age of OB Associations')
    plt.xlabel('Distance (pc)')
    plt.ylabel('Age (Myr)')
    plt.grid(True)
    plt.savefig(f'{FOLDER_OUTPUT}/{filename}')
    plt.close()


def plot_associations(glon, heliocentric_distance, filename):
    """ Plot the distribution of known associations in the Galactic plane together with the spiral arms medians
    
    Args:
        glon: array. Galactic longitude of associations to be plotted
        heliocentric_distance: array. Heliocentric distance of associations to be plotted
        filename: str. Name of the file to save the plot
    
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


def gaussian(x, a, b, c): # Gaussian function for fitting
    return a * np.exp(-((x - b)**2) / (2 * c**2))


def gaussian_fixed_peak(x, b, c): # Gaussian function for fitting with fixed peak
    return avg_hist * np.exp(-((x - b)**2) / (2 * c**2))


def plot_distance_hist(heliocentric_distance, filename, wright=False):
    """ Plot the histogram of distances of OB associations and fit the data to a Gaussian function

    Args:
        heliocentric_distance: array. Heliocentric distances of the associations
        filename: str. Name of the file to save the plot
        wright: bool. If True, makes a piecewise fitter curve. Constant for r > 2.5 kpc and fits the data for r > 2 kpc to a Gaussian function
    
    Returns:
        None. Saves the plot"""
    endpoint = 5000 # max radial distance in pc for the histogram
    step = 500  # stepsize for the histogram in pc
    num = endpoint // step + 1 # number of bins
    bins = np.linspace(start=0, stop=endpoint, num=num, endpoint=True)
    hist, _ = np.histogram(heliocentric_distance, bins=bins)
    hist_central_x_val = bins[:-1] + step / 2 # central x values for each bin
    bins, hist_central_x_val = bins / 1000, hist_central_x_val / 1000 # convert to kpc
    global avg_hist # make global so that it can be used in the fitting function
    avg_hist = hist[hist_central_x_val <= 2.5] # pick out the associations for r < 2.5 kpc
    avg_hist = np.mean(avg_hist) # average the associatins for r < 2.5 kpc
    # Make the histogram    
    plt.figure(figsize=(10, 6))
    plt.hist(heliocentric_distance / 1000, bins=bins, color='green', alpha=0.7)
    # Fit the entire dataset to the Gaussian function
    params, cov = curve_fit(gaussian, hist_central_x_val, hist, p0 = [max(hist), np.mean(hist_central_x_val), np.std(hist_central_x_val)])
    x_interpolated = np.linspace(0, endpoint, 500) / 1000
    y_fit = gaussian(x_interpolated, *params)
    plt.plot(x_interpolated, y_fit, label='Fitted Gaussian', color='purple')
    if wright:
        # Fit the dataset for r > 2 kpc to the Gaussian function
        filtered_indices = hist_central_x_val > 2
        params, cov = curve_fit(gaussian_fixed_peak, hist_central_x_val[filtered_indices], hist[filtered_indices], p0=[np.mean(hist_central_x_val[filtered_indices]), np.std(hist_central_x_val[filtered_indices])])
        # Generate points for the piecewise fitted curve
        x_gauss_2 = np.linspace(2, endpoint, 500) / 1000  # Gaussian part
        y_gauss_2 = gaussian_fixed_peak(x_gauss_2, *params) # Gaussian part
        max_y_index = np.argmax(y_gauss_2) # To find the dividing point between the two parts
        x_1 = np.linspace(0, x_gauss_2[max_y_index], 500)   # Average part, completelly flat
        y_1 = np.ones(500) * avg_hist   # Average part, completelly flat
        x_gauss_2 = np.concatenate((x_1, x_gauss_2[max_y_index:]))  # Combine the two parts
        y_gauss_2 = np.concatenate((y_1, y_gauss_2[max_y_index:]))  # Combine the two parts
        plt.plot(x_gauss_2, y_gauss_2, label='Fitted Gaussian 2', color='yellow', zorder=10)
        # Denote the average value for r < 2.5 kpc
        x_avg = np.linspace(x_gauss_2[max_y_index], 5, 500)
        y_avg = np.ones(500) * avg_hist
        plt.plot(x_avg, y_avg, color='black', linestyle='--', label='Average for r < 2.5 kpc', zorder=9)
    # Complete the plot
    plt.title('Histogram of Distances')
    plt.xlabel('Distance (kpc)')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(f'{FOLDER_OUTPUT}/{filename}')
    plt.close()


def data_wright(filter_data=False):
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
    plot_distance_hist(wright_distance, filename=f'wright_distance_hist_mask_{filter_data}.pdf', wright=True)
    plot_associations(wright_glon, wright_distance, filename=f'wright_associations_arms_mask_{filter_data}.pdf')
    plot_age_distance(wright_distance, wright_age, filename=f'wright_age_distance_mask_{filter_data}.pdf')


def main():
    file_path = f'{FOLDER_OBS_DATA}/Overview of know OB associations.xlsx' 
    data = pd.read_excel(file_path)
    #print(data.describe()) # Get basic statistics
    plot_age_distance(data['Distance (pc)'], data['Age(Myr)'], filename='my_data_age_distance.pdf')
    plot_distance_hist(data['Distance (pc)'], filename='my_data_distance_hist.pdf')
    plot_associations(data['l (deg)'], data['Distance (pc)'], filename='my_data_associations.pdf')
    data_wright(True)
    data_wright(False)


if __name__ == '__main__':
    main()