import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Load data from the text file
data = np.loadtxt('src/N+.txt')
print(data.shape)
print(data[:, 1].shape)
print(data[:, 1])
data = data[np.abs(data[:, 1]) <= 5]
print(data[:, 1])

print(data.shape)
# Extract relevant columns
lon = data[:, 0]
lat = data[:, 1]
intensity = data[:, 2] # units of MJy/sr
weight = data[:, 3]
""" intensity *= 1e6 # convert to Jy/sr
intensity *= 1e-26 # convert to W/m^2/sr
intensity *= 1e9 # convert to nW/m^2/sr
intensity = intensity * weight """
#intensity = intensity *weight


""" def stupid_skymap():
    # Set the resolution (adjust nside based on your needs)
    nside = 22
    # Calculate the total number of pixels
    npix = hp.nside2npix(nside)
    print("Number of pixels in the skymap:", npix)
    # Convert longitude and latitude to HEALPix indices
    pixels = hp.ang2pix(nside, lon, lat, lonlat=True)
    # Create a HEALPix map using intensity values
    healpix_map = np.zeros(hp.nside2npix(nside))
    healpix_map[pixels] = intensity
    # Plot the HEALPix map
    hp.mollview(healpix_map, title='Your Sky Map Title', cmap='viridis')
    # Show the plot
    plt.show() """
print(len(lon), len(lat), len(intensity), len(weight))

def sum_pairwise(a):
    paired_data = a.reshape(-1, 2)
    # Sum along the specified axis (axis=1 sums along columns)
    result = np.sum(paired_data, axis=1)
    return result

# Define bin edges with a bin width of 5 degrees
bin_width = 5
""" 
bin_edge_1_2_5 = np.array([180, 180-2.5])
bin_edge_1 = np.arange(180-2.5, 2.5, -bin_width)
bin_edge_1_centre = np.array([2.5, 0])
bin_edge_2_centre = np.array([0, 360 - 2.5])
bin_edge_2 = np.arange(360 - 2.5, 180 - 2.5, -bin_width)
bin_edge_2_2_5 = np.array([180 - 2.5, 180])
bin_edges = np.arange(0, 365, bin_width)
bin_edges = np.concatenate((bin_edge_1_2_5, bin_edge_1, bin_edge_1_centre, bin_edge_2_centre, bin_edge_2, bin_edge_2_2_5))
print(len(bin_edges)) """
bin_edges = np.arange(0, 362.5, 2.5)
print(bin_edges)

# Use histogram to compute the binned intensity values and bin centers
hist, bin_edges = np.histogram(lon, bins=bin_edges, weights=intensity) # if a longitude is in the bin, add the intensity to the bin
hist_num_long_per_bin, bin_edges_num_long_per_bin = np.histogram(lon, bins=bin_edges)
hist = hist / hist_num_long_per_bin
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot the binned data
#plt.bar(bin_centers, hist, width=bin_width/2, align='center', edgecolor='black')
# plt.bar uses x-position. Since I need custom x-axis, use x-ticks to set the x-axis
# one plt. bar for the leftmost and rightmost bins
x_1 = 0 # 180 degrees
x_2 = np.arange(2.5, 360 - 2.5, 5) # 180-2.5, down to 0, up to 180 degrees again
x_3 = 360 - 2.5 # 180 degrees
print("x2 length: ", len(x_2))
print("x1: ", x_1)
print("x2: ", x_2)
print("x3: ", x_3)
data_centre_left = hist[0]
data_left = sum_pairwise(hist[1:71]) / 2 
data_left_edge = hist[71]
data_right_edge = hist[72]
data_right = sum_pairwise(hist[73:-1]) / 2
data_centre_right = hist[-1]

print(data_left.shape, data_right.shape, data_centre_left.shape)
print(data_centre_left)
print(len(data_left), len(data_right))
print(data_left[-1], data_left_edge)
print(1+1 + len(data_left) + 1+1 + len(data_right))
print(len(hist))
print(len(bin_edges))

data_central= np.concatenate((data_left[::-1], [(data_centre_left + data_centre_right)/2], data_right[::-1]))
plt.bar(x_1, data_left_edge, width=bin_width/2, align='edge', edgecolor='black', color='blue')
plt.bar(x_2, data_central, width=bin_width, align='edge', edgecolor='black', color='blue')
plt.bar(x_3, data_right_edge, width=bin_width/2, align='edge', edgecolor='black', color='blue')
x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
plt.xticks(np.linspace(0, 360, 13), x_ticks)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
plt.xlabel('Galactic longitude (degrees)')
plt.ylabel('Line intensity in some units')
plt.title("N+ line intensity vs Galactic longitude")
plt.show()