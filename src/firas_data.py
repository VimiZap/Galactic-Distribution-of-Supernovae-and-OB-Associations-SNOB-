import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Load data from the text file
data = np.loadtxt('src/N+.txt')
#data = data[np.abs(data[:, 1]) <= 5]
print("shape data", data.shape)
# Extract relevant columns

""" intensity *= 1e6 # convert to Jy/sr
intensity *= 1e-26 # convert to W/m^2/sr
intensity *= 1e9 # convert to nW/m^2/sr
intensity = intensity * weight """

#print(len(lon), len(lat), len(intensity), len(weight))


def sum_pairwise(a):
    paired_data = a.reshape(-1, 2)
    # Sum along the specified axis (axis=1 sums up each row)
    result = np.sum(paired_data, axis=1)
    return result


def calc_hist(data):
    long = data[:, 0]
    lat = data[:, 1]
    intensity = data[:, 2] # units of MJy/sr
    intensity *= 1e6 * 1e-26 * 1e9 * 1.463*1e12 # convert to nW/m^2/sr. 1.463e12 is the frequency of the N+ line in Hertz
    weight = data[:, 3]
    bin_edges_long = np.arange(0, 362.5, 2.5) # will end at 360. No data is left out. 145 bin edges
    bin_edges_lat = np.arange(-89.5, 90, 1) # will end at 89.5. No data is left out. 180 bin edges
    # Use histogram to compute the binned intensity values and bin centers
    hist, _, _ = np.histogram2d(long, lat, bins=(bin_edges_long, bin_edges_lat), weights=intensity, density=False)
    print("Shape of hist: ", hist.shape)
    num_counts_per_bin, _, _ = np.histogram2d(long, lat, bins=(bin_edges_long, bin_edges_lat))
    print("sum of num_counts_per_bin", np.sum(num_counts_per_bin))
    # Avoid division by zero: where count is zero, set it to 1 (will result in zero intensity for these bins)
    num_counts_per_bin[num_counts_per_bin == 0] = 1
    hist = hist / num_counts_per_bin
    hist = np.sum(hist, axis=1) #/ len(hist[1])
    print("Shape of hist after sum along axis=1: ", hist.shape)
    return hist

def calc_hist_2(data):
    long = data[:, 0]
    lat = data[:, 1]
    intensity = data[:, 2] # units of MJy/sr
    intensity *= 1e6 * 1e-26 * 1e9 * 1.463*1e12 # convert to nW/m^2/sr. 1.463e12 is the frequency of the N+ line in Hertz
    weight = data[:, 3]
    bin_edges_long = np.arange(0, 362.5, 2.5) # will end at 360. No data is left out. 145 bin edges
    bin_edges_lat = np.arange(-89.5, 90, 1) # will end at 89.5. No data is left out. 180 bin edges
    hist, bin_edges = np.histogram(long, bins=bin_edges_long, weights=intensity) # if a longitude is in the bin, add the intensity to the bin
    hist_num_long_per_bin, bin_edges_num_long_per_bin = np.histogram(long, bins=bin_edges)
    print(hist_num_long_per_bin)
    print(np.sum(hist_num_long_per_bin))
    hist = hist / hist_num_long_per_bin
    return hist

def plot_hist_data(hist):
    # partition the data to be plotted in desired format
    middle = int(len(hist)/2)
    data_centre_left = hist[0]
    data_left = sum_pairwise(hist[1:middle-1]) / 2
    data_left_edge = hist[middle-1]
    data_right_edge = hist[middle]
    data_edge = (data_right_edge + data_left_edge) / 2
    data_right = sum_pairwise(hist[middle+1:-1]) / 2
    data_centre_right = hist[-1]
    data_centre = (data_centre_left + data_centre_right)/2

    # Create bin_values
    data_central = np.concatenate(([data_edge], data_left[::-1], [data_centre], data_right[::-1], [data_edge]))
    # Create bin_edges
    bin_edges_central = np.arange(2.5, 360, 5)
    bin_edges = np.concatenate(([0], bin_edges_central, [360]))
    # Plot using stairs
    plt.stairs(values=data_central, edges=bin_edges, fill=False, color='black')
    # Set up the plot
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3))
    plt.xlabel('Galactic longitude (degrees)')
    plt.xlim(0, 360)
    plt.ylabel('Line intensity in nW m$^{-2}$ sr$^{-1}$')
    plt.title("N+ line intensity vs Galactic longitude")
    # Save the plot
    plt.savefig("output/firas_data_hist_contour.png", dpi=1200)
    plt.close()

hist = calc_hist_2(data)
plot_hist_data(hist)