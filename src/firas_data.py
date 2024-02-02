import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def sum_pairwise(a):
    paired_data = a.reshape(-1, 2)
    # Sum along the specified axis (axis=1 sums up each row)
    result = np.sum(paired_data, axis=1)
    return result


def rearange_data(data):
    # rearange data to be plotted in desired format. Also does the summation
    middle = int(len(data)/2)
    data_centre_left = data[0]
    data_left = sum_pairwise(data[1:middle-1])
    data_left_edge = data[middle-1]
    data_right_edge = data[middle]
    data_edge = (data_right_edge + data_left_edge)
    data_right = sum_pairwise(data[middle+1:-1])
    data_centre_right = data[-1]
    data_centre = (data_centre_left + data_centre_right)
    rearanged_data = np.concatenate(([data_edge], data_left[::-1], [data_centre], data_right[::-1], [data_edge]))
    return rearanged_data


def calc_hist_1d(data):
    #1D binning of data
    long = data[:, 0]
    intensity = data[:, 2] # units of MJy/sr
    # set negative values in intensity to zero
    intensity[intensity < 0] = 0
    intensity *= 1e6 * 1e-26 * 1e9 * 1.463*1e12 # convert to nW/m^2/str. 1.463e12 is the frequency of the N+ line in Hertz
    bin_edges_long = np.arange(0, 362.5, 2.5) # will end at 360. No data is left out. 145 bin edges
    hist, _ = np.histogram(long, bins=bin_edges_long, weights=intensity) # if a longitude is in the bin, add the intensity to the bin
    hist_num_long_per_bin, _ = np.histogram(long, bins=bin_edges_long)
    # Rearange data to be plotted in desired format
    rearanged_hist = rearange_data(hist)
    rearanged_hist_num_long_per_bin = rearange_data(hist_num_long_per_bin)
    rearanged_hist_num_long_per_bin[rearanged_hist_num_long_per_bin == 0] = 1
    hist = rearanged_hist / rearanged_hist_num_long_per_bin
    return hist


def plot_hist_data(hist, filename):
    # Create bin_edges
    bin_edges_central = np.arange(2.5, 360, 5)
    bin_edges = np.concatenate(([0], bin_edges_central, [360]))
    # Plot using stairs
    print("len(hist)", len(hist))
    print("len(bin_edges)", len(bin_edges))
    plt.stairs(values=hist, edges=bin_edges, fill=False, color='black')
    # Set up the plot
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3))
    plt.xlabel('Galactic longitude (degrees)')
    plt.xlim(0, 360)
    plt.ylabel('Line intensity in nW m$^{-2}$ sr$^{-1}$')
    plt.title("N+ line intensity vs Galactic longitude")
    # Save the plot
    plt.savefig(filename, dpi=1200)
    plt.close()


def scatter_fixen_data(data):
    long = data[:, 0]
    lat = data[:, 1]
    intensity = data[:, 2] # units of MJy/sr
    plt.scatter(long, lat, c=intensity, cmap='viridis', s=10)
    plt.show()


def main():
    # Load data from the text file
    data = np.loadtxt('src/N+.txt')
    print("shape data", data.shape)
    print("number of datapoints with negative intensity:", len(data[data[:, 2] < 0]))
    print("number of datapoints with latitude < |5|:", len(data[np.abs(data[:, 1]) <= 5]))
    data = data[np.abs(data[:, 1]) <= 5]    
    hist_1 = calc_hist_1d(data)
    plot_hist_data(hist_1, "output/firas_data_final_estimate.png")
    #scatter_fixen_data(data)


if __name__ == "__main__":
    main()