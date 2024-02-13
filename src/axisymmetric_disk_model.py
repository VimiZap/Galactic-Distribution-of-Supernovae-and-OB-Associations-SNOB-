import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import gc
import observational_data.firas_data as firas_data
import logging

# constants
h = 2.5                # kpc, scale length of the disk. The value Higdon and Lingenfelter used
r_s = 7.6 #8.178            # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
rho_min =  3 # 0.39 * r_s    # kpc, minimum distance from galactic center to bright H 2 regions. Evaluates to 3.12 kpc
rho_max = 10 # 1.30 * r_s    # kpc, maximum distance from galactic center to bright H 2 regions. Evaluates to 10.4 kpc
sigma = 0.15            # kpc, scale height of the disk
measured_nii_30_deg = 0.00011711056373558678 # erg/s/cm²/sr measured N II 205 micron line intensity at 30 degrees longitude. Retrieved from the FIRAS data with the function firas_data.find_firas_intensity_at_central_long(30)
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
# note that the authors where this formula came from used different numbers, and got a value of 47 kpc^2.
# With these numbers, we get a value of 23.418 kpc^2. Compare this to the area of pi*(rho_max**2 - rho_min**2) = 309.21 kpc^2
a_d = 2*np.pi*h**2 * ((1+rho_min/h)*np.exp(-rho_min/h) - (1+rho_max/h)*np.exp(-rho_max/h)) 
print("a_d = ", a_d)
kpc = 3.08567758e21    # 1 kpc in cm


def axisymmetric_disk_model(rho): ###### could also move to utilities.py
    """
    Args:
        rho: distance from the Galactic center
        h: scale length of the disk
    Returns:
        density of the disk at a distance rho from the Galactic center
    """
    return np.exp(-rho/h)


def calc_modelled_intensity(b_max = 5):
    logging.info("Running calc_modelled_intensity()")
    # load the coordinate data:
    num_lats = len(np.lib.format.open_memmap('output/galaxy_data/latitudes.npy'))
    num_rads = len(np.lib.format.open_memmap('output/galaxy_data/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap('output/galaxy_data/longitudes.npy'))
    
    db = np.load('output/galaxy_data/db.npy')
    dr = np.load('output/galaxy_data/dr.npy')
    dl = np.load('output/galaxy_data/dl.npy')
    
    latitudinal_cosinus = np.lib.format.open_memmap('output/galaxy_data/latitudinal_cosinus.npy')
    height_distribution_values = np.lib.format.open_memmap('output/galaxy_data/height_distribution_values.npy')
    common_multiplication_factor =  dr * latitudinal_cosinus * height_distribution_values/ (4 * np.pi * np.radians(b_max * 2) * np.radians(5) * a_d * kpc**2) 
    common_multiplication_factor = np.reshape(common_multiplication_factor, (num_rads, num_longs, num_lats)) * db[np.newaxis, np.newaxis, :] # reshaping to facilitate the multiplication with non-uniform latitudinal increments db
    common_multiplication_factor = common_multiplication_factor.ravel() #unraveling so that we can multiply with the interpolated densities
    del latitudinal_cosinus, db, dr
    gc.collect()
    
    rho = np.lib.format.open_memmap('output/galaxy_data/rho_coords_galaxy.npy')
    logging.info("Data loaded. Calculating modelled intensity")
    intensities = axisymmetric_disk_model(rho) * common_multiplication_factor # 3D array with the modelled emissivity for each value of r, l and b
    logging.info("Modelled intensity calculated. Removing values outside the bright H 2 regions and normalizing to the measured value at 30 degrees longitude")
    mask = (rho > rho_max) | (rho < rho_min)
    # Set values in intensities to zero where the mask is True
    intensities[mask] = 0
    intensities = np.sum(intensities.reshape(num_rads, num_longs, num_lats), axis=(0, 2)) # sum over the radial and latitudinal axis
    # Now normalize the modelled intensity to the measured value at 30 degrees longitude in the FIRAS data
    longitudes = np.lib.format.open_memmap('output/galaxy_data/longitudes.npy')
    abs_diff = np.abs(longitudes - np.radians(30))  # Calculate the absolute differences between the 30 degrees longitude and all elements in longitudes
    closest_index = np.argmin(abs_diff) # Find the index of the element with the smallest absolute difference
    modelled_value_30_degrees = intensities[closest_index] # Retrieve the closest value from the integrated spectrum
    luminosity_axisymmetric = measured_nii_30_deg / modelled_value_30_degrees # Calculate the normalization factor
    intensities *= luminosity_axisymmetric # normalize the modelled emissivity to the measured value at 30 degrees longitude
    # Note: for comparison reasons with Higdon and Lingenfelter, I am not using a running average on the intensities, as it smoothen the spikes
    logging.info("Saving the modelled intensity and the normalization factor")
    np.save('output/galaxy_data/luminosity_axisymmetric.npy', luminosity_axisymmetric)
    np.save('output/galaxy_data/intensities_axisymmetric.npy', intensities) # normalize the modelled emissivity to the measured value at 30 degrees longitude
    return


def plot_axisymmetric():
    longitudes = np.lib.format.open_memmap('output/galaxy_data/longitudes.npy')
    intensities = np.lib.format.open_memmap('output/galaxy_data/intensities_axisymmetric.npy')
    #longitudes, integrated_spectrum = integral_axisymmetric()
    luminosity = np.load('output/galaxy_data/luminosity_axisymmetric.npy')
    fig, ax = plt.subplots()
    firas_data.add_firas_data_to_plot(ax)
    ax.plot(np.linspace(0, 360, len(longitudes)), intensities)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    ax.set_xticks(np.linspace(0, 360, 13))
    ax.set_xticklabels(x_ticks)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel("Galactic longitude l (degrees)")
    ax.set_ylabel("Modelled emissivity")
    ax.set_title("Modelled emissivity of the Galactic disk")
    ax.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_z$ = {sigma} kpc', transform=ax.transAxes, fontsize=8, color='black')
    ax.text(0.02, 0.9, fr'NII Luminosity = {luminosity:.2e} erg/s', transform=ax.transAxes, fontsize=8, color='black')
    ax.text(0.02, 0.85, fr'{rho_min:.2e}  $\leq \rho \leq$ {rho_max:.2e} kpc', transform=ax.transAxes, fontsize=8, color='black')
    fig.savefig("src/data_products/modelled_emissivity_axisymmetric.png", dpi=1200)  # save plot in the output folder
    plt.close(fig)


def plot_galacic_centric_distribution():
    r = np.linspace(rho_min, rho_max, 1000) # kpc, distance from the Galactic center
    plt.plot(r, axisymmetric_disk_model(r, h))
    plt.xlabel("Distance from the Galactic center (kpc)")
    plt.ylabel("Density")
    plt.title("Axisymmetric disk model with scale length h = " + str(h) + " kpc")
    plt.savefig("output/axisymmetric_disk_model.png")     # save plot in the output folder
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO) 
    calc_modelled_intensity()
    plot_axisymmetric()


if __name__ == "__main__":
    main()