import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
import logging
import utilities as ut


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
    rearanged_hist = ut.rearange_data(hist)
    rearanged_hist_num_long_per_bin = ut.rearange_data(hist_num_long_per_bin)
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


def plot_data_from_fixsen():
    # Load data from the text file
    data = np.loadtxt('src/observational_data/N+.txt')
    print("shape data", data.shape)
    print("number of datapoints with negative intensity:", len(data[data[:, 2] < 0]))
    print("number of datapoints with latitude < |5|:", len(data[np.abs(data[:, 1]) <= 5]))
    data = data[np.abs(data[:, 1]) <= 5]    
    hist_1 = calc_hist_1d(data)
    plot_hist_data(hist_1, "src/observational_data/firas_data_final_estimate.png")
    #scatter_fixen_data(data)


def create_bin_edges_from_central_values(central_values, bin_half_width):
    first_edge = central_values[0]
    last_edge = central_values[-1]
    bin_edges = []
    for i in range(len(central_values) - 1):
            bin_edges.append(central_values[i + 1] - bin_half_width)
    bin_edges = np.concatenate(([first_edge], bin_edges, [last_edge]))
    return bin_edges


def firas_data_for_plotting():
    fits_file = fits.open('src/observational_data/lambda_firas_lines_1999_galplane.fits')
    #fits_file.info()
    # grab the data from the 12th HDU
    data_hdu = fits_file[12] 
    data = data_hdu.data
    # extract the data
    line_id = data['LINE_ID']
    gal_lon = data['GAL_LON'][0] + 180 # add 180 to get only positive values, from 0 to 360. Contains the central values of the bins
    bin_edges_line_flux = create_bin_edges_from_central_values(gal_lon, 2.5) # create bin-edges for the plotting
    bin_centre_line_flux = np.concatenate(([gal_lon[0] + 2.5/2], gal_lon[1:-1], [gal_lon[-1] - 2.5/2])) # create bin-centres for the error-plotting
    line_flux = data['LINE_FLUX'][0] *1e-4 * 1e-9 *1e7  # convert from nW/m^2/str to erg/s/cm²/sr.
    line_flux_error = data['LINE_FLERR'][0] * 1e-4 * 1e-9 * 1e7 * 2  # convert from nW/m^2/str to erg/s/cm²/sr. Multiply by 2 to get 2-sigma error
    return bin_edges_line_flux, bin_centre_line_flux, line_flux[::-1], line_flux_error[::-1] # reverse the order of the data to match the longitudes


def find_firas_intensity_at_central_long(long):
    """ Find the intensity of the N+ line at a given longitude in the FIRAS data. The intensity is given in erg/s/cm²/sr.

    Args:
        long: int, the longitude in degrees. Valid values are in the range -180 to 180, with increments in 5 degrees.

    Returns:
        float, the intensity of the N+ line at the given longitude.
    """
    folder = 'src/observational_data'
    fits_file = fits.open(f'{folder}/lambda_firas_lines_1999_galplane.fits')
    data_hdu = fits_file[12] 
    data = data_hdu.data
    gal_lon = data['GAL_LON'][0] 
    line_flux = data['LINE_FLUX'][0] *1e-4 * 1e-9 *1e7  # convert from nW/m^2/str to erg/s/cm²/sr.
    if long not in gal_lon:
        print("The longitude is not in the data. Valid values are in the range -180 to 180, with increments in 5 degrees.")
        return
    index = np.where(gal_lon == long)
    logging.info("The intensity at longitude", long, " degrees is", line_flux[index][0], "erg/s/cm²/sr.")
    return line_flux[index][0]


def plot_firas_nii_line():
    folder_output = 'src/observational_data'
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data_for_plotting()
    plt.figure(figsize=(10, 6))
    plt.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    plt.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 360, 13), x_ticks)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(3)) 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Line intensity in erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$")
    plt.title("Modelled intensity of the Galactic disk")
    plt.xlim(0, 360)
    plt.savefig(f'{folder_output}/firas_data_NII_line.png', dpi=1200)
    plt.close()


def add_firas_data_to_plot(ax):
    """ Add the FIRAS data to an existing plot. ax's x-range should be 0 to 360.

    Args:
        ax: matplotlib axis object

    Returns:
        None
    """
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data_for_plotting()
    ax.stairs(values=line_flux, edges=bin_edges_line_flux, fill=False, color='black')
    ax.errorbar(bin_centre_line_flux, line_flux, yerr=line_flux_error,fmt='none', ecolor='black', capsize=0, elinewidth=1)

def main():
    #plot_firas_nii_line()
    #plot_data_from_fixsen()
    find_firas_intensity_at_central_long(30)


if __name__ == "__main__":
    main()