import numpy as np
import matplotlib.pyplot as plt

# constants
h = 2.5                 # kpc, scale length of the disk. The value Higdon and Lingenfelter used
r_s = 7.6               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
rho_min = 0.39 * r_s    # kpc, minimum distance from galactic center to bright H 2 regions. Evaluates to 3.12 kpc
rho_max = 1.30 * r_s    # kpc, maximum distance from galactic center to bright H 2 regions. Evaluates to 10.4 kpc
sigma = 0.15            # kpc, scale height of the disk
total_galactic_n_luminosity = 1.85e40       #total galactic N 2 luminosity in erg/s
measured_nii = 1.175e-4 # erg/s/cm^2/sr, measured N II 205 micron line intensity. Estimated from the graph in the paper
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
# note that the authors where this formula came from used different numbers, and got a value of 47 kpc^2.
# With these numbers, we get a value of 22.489 kpc^2. Compare this to the area of pi*(rho_max**2 - rho_min**2) = 309.21 kpc^2
a_d = 2*np.pi*h**2 * ((1+rho_min/h)*np.exp(-rho_min/h) - (1+rho_max/h)*np.exp(-rho_max/h)) 


def axisymmetric_disk_model(rho): 
    """
    Args:
        rho: distance from the Galactic center
        h: scale length of the disk
    Returns:
        density of the disk at a distance rho from the Galactic center
    """
    return np.exp(-rho/h)


def rho(r, l, b):
    """
    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        distance from the Galactic center to the star/ a point in the Galaxy
    """
    return np.sqrt((r * np.cos(b))**2 + r_s**2 - 2 * r_s * r * np.cos(b) * np.cos(l)) # kpc, distance from the Sun to the star/ spacepoint


def height_distribution(z): # z is the height above the Galactic plane
    """
    Args:
        z: height above the Galactic plane
    Returns:
        height distribution of the disk at a height z above the Galactic plane
    """
    return np.exp(-0.5 * z**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)


def modelled_emissivity_axisymmetric(r, l, b):
    """
    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        modelled emissivity of the Galactic disk with the axisymmetric disk model
    """
    return axisymmetric_disk_model(rho(r, l, b)) * height_distribution(r*np.sin(b)) * total_galactic_n_luminosity / a_d


def integrand_axisymmetric(r, l, b):
    """
    Args:
        r: distance from the Sun to the star/ a point in the Galaxy
        l: Galactic longitude, radians
        b: Galactic latitude, radians
    Returns:
        integrand of the integral to calculate the modelled emissivity of the Galactic disk for the axisymmetric disk model
    """
    return modelled_emissivity_axisymmetric(r, l, b)  * np.cos(b) / (4 * np.pi)


def integral_axisymmetric():
    """
    Returns:
        modelled disk intensity as a function of Galactic longitude l
    """
    dr = 0.1   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.01   # increments in db (degrees):

    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latidue_range = np.radians(3.5)
    latitudes = np.arange(-latidue_range, latidue_range, db)

    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 179, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))

    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(0, r_s + rho_max, dr) #r_s + rho_max is the maximum distance from the Sun to the outer edge of the Galaxy
    integral = [] # list to store the values of the integral for each value of l
    for l in longitudes:
        value = 0
        for b in latitudes:
            for r in radial_distances:
                if rho(r, l, b) > rho_max:
                    break # we are outside of the Galaxy and are thus done integrating along this line of sight
                elif rho(r, l, b) < rho_min:
                    continue # we are inside the Galaxy, but not within the bright H 2 regions, so we can skip this point
                else:
                    value += integrand_axisymmetric(r, l, b) * dr
        #print(r, l, b, value*db)
        integral.append(value * db) # multiply by db to get the integral over the latitude

    # when done with integrating over r and b, return the value as we are not integrating over l
    return longitudes, integral / (np.radians(1)) #* np.radians(5)) # devide by delta-b and delta-l in radians, respectively, for the averaging the paper mentions


def plot_axisymmetric():
    longitudes, integrated_spectrum = integral_axisymmetric()
    abs_diff = np.abs(longitudes - np.radians(30))  # Calculate the absolute differences between the 30 degrees longitude and all elements in longitudes
    closest_index = np.argmin(abs_diff) # Find the index of the element with the smallest absolute difference
    modelled_value_30_degrees = integrated_spectrum[closest_index] # Retrieve the closest value from the integrated spectrum
    normalization_factor = measured_nii / modelled_value_30_degrees # Calculate the normalization factor
    #print(np.degrees(longitudes[closest_index]), modelled_value_30_degrees, normalization_factor)
    integrated_spectrum = integrated_spectrum * normalization_factor # normalize the modelled emissivity to the measured value at 30 degrees longitude
    plt.plot(np.linspace(0, 100, len(longitudes)), integrated_spectrum)
    print(np.sum(integrated_spectrum))
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    plt.savefig("output/modelled_emissivity_axisymmetric.png")  # save plot in the output folder
    plt.show()


def plot_galacic_centric_distribution():
    r = np.linspace(rho_min, rho_max, 1000) # kpc, distance from the Galactic center
    plt.plot(r, axisymmetric_disk_model(r, h))
    plt.xlabel("Distance from the Galactic center (kpc)")
    plt.ylabel("Density")
    plt.title("Axisymmetric disk model with scale length h = " + str(h) + " kpc")
    plt.savefig("output/axisymmetric_disk_model.png")     # save plot in the output folder
    plt.show()


def main():
    plot_axisymmetric()


if __name__ == "__main__":
    main()