import numpy as np
import matplotlib.pyplot as plt
import time

# constants
h = 2.5                # kpc, scale length of the disk. The value Higdon and Lingenfelter used
r_s = 8.178            # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
rho_min =  3 # 0.39 * r_s    # kpc, minimum distance from galactic center to bright H 2 regions. Evaluates to 3.12 kpc
rho_max = 11 # 1.30 * r_s    # kpc, maximum distance from galactic center to bright H 2 regions. Evaluates to 10.4 kpc
sigma = 0.15            # kpc, scale height of the disk
total_galactic_n_luminosity = 1 #1.85e40       #total galactic N 2 luminosity in erg/s
measured_nii = 1.175e-4 # erg/s/cm^2/sr, measured N II 205 micron line intensity. Estimated from the graph in the paper
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
# note that the authors where this formula came from used different numbers, and got a value of 47 kpc^2.
# With these numbers, we get a value of 23.418 kpc^2. Compare this to the area of pi*(rho_max**2 - rho_min**2) = 309.21 kpc^2
a_d = 2*np.pi*h**2 * ((1+rho_min/h)*np.exp(-rho_min/h) - (1+rho_max/h)*np.exp(-rho_max/h)) 
print("a_d = ", a_d)
kpc = 3.08567758e21    # 1 kpc in cm
#a_d = a_d * kpc**2     # convert a_d to cm^2


def running_average(data, window_size):
   array_running_averaged = []
   delta = int((window_size)//2)
   print("running average: ", window_size, delta)
   for i in range(len(data)):
      if i-delta < 0:
         val = np.sum(data[-delta + i:]) + np.sum(data[:delta + i + 1])
         array_running_averaged.append(val)
      elif i+delta >= len(data):
         val = np.sum(data[i-delta:]) + np.sum(data[:delta + i - len(data) + 1])
         array_running_averaged.append(val)
      else:
         array_running_averaged.append(np.sum(data[i-delta:i+delta + 1]))
   return np.array(array_running_averaged)


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
    dr = 0.01   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.5   # increments in db (degrees):

    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    #latidue_range = np.radians(3.5)
    latitudes = np.radians(np.arange(-3.5, 3.5 + db, db)) # the +db is to include the last value in the range

    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))

    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(dr, r_s + rho_max + dr, dr) #r_s + rho_max is the maximum distance from the Sun to the outer edge of the Galaxy
    integral = [] # list to store the values of the integral for each value of l
    densities_skymap = np.zeros((len(longitudes) + 1, len(latitudes) + 1))
    for l in range(len(longitudes)):
        value = 0
        for b in range(len(latitudes)):
            for r in radial_distances:
                long = longitudes[l]
                lat = latitudes[b]
                if rho(r, long, lat) > rho_max:
                    break # we are outside of the Galaxy and are thus done integrating along this line of sight
                elif rho(r, long, lat) < rho_min:
                    #print("rho: ", rho(r, l, b))
                    continue # we are inside the Galaxy, but not within the bright H 2 regions, so we can skip this point
                else:
                    #print(integrand_axisymmetric(r, l, b) * dr * db)
                    lbr_value = integrand_axisymmetric(r, long, lat) * dr * db
                    value += lbr_value
                    densities_skymap[l + 1, b + 1] += lbr_value
        integral.append(value) # multiply by db to get the integral over the latitude
    print("len latitudes: ", len(latitudes))
    multiplication_constant = 1 / kpc**2 
    integral = np.array(integral) * multiplication_constant 
    abs_diff = np.abs(longitudes - np.radians(30))  # Calculate the absolute differences between the 30 degrees longitude and all elements in longitudes
    closest_index = np.argmin(abs_diff) # Find the index of the element with the smallest absolute difference
    modelled_value_30_degrees = integral[closest_index] # Retrieve the closest value from the integrated spectrum
    galactic_luminocity = measured_nii / modelled_value_30_degrees # Calculate the normalization factor
    print(modelled_value_30_degrees, galactic_luminocity, galactic_luminocity/1.85e40)
    integral *= galactic_luminocity # normalize the modelled emissivity to the measured value at 30 degrees longitude
    # skymap
    densities_skymap[1:, 1:] = densities_skymap[1:, 1:] * multiplication_constant * galactic_luminocity
    filepath = "output/long_lat_skymap_axisymmetric.txt"
    densities_skymap[1:, 0] = longitudes
    densities_skymap[0, 1:] = latitudes
    np.savetxt(filepath, densities_skymap)
    print("len latitudes: ", len(latitudes))
    # running average:
    """ window_size = 5 / dl # 5 degrees in divided by the increment in degrees for the longitude. This is the window size for the running average, number of points
    integral = running_average(integral, window_size) / window_size #divide by delta-l in radians for the averaging the paper mentions """
    
    #integrated_spectrum = integrated_spectrum/kpc**2 # convert from cm^-2 to kpc^-2
    
    return longitudes, integral

def integral_axisymmetric_old():
    """
    Returns:
        modelled disk intensity as a function of Galactic longitude l
    """
    dr = 0.01   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.5   # increments in db (degrees):

    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    #latidue_range = np.radians(3.5)
    latitudes = np.radians(np.arange(-3.5, 3.5 + db, db)) # the +db is to include the last value in the range

    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))

    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(dr, r_s + rho_max + dr, dr) #r_s + rho_max is the maximum distance from the Sun to the outer edge of the Galaxy
    integral = [] # list to store the values of the integral for each value of l
    for l in longitudes:
        value = 0
        for b in latitudes:
            for r in radial_distances:
                if rho(r, l, b) > rho_max:
                    break # we are outside of the Galaxy and are thus done integrating along this line of sight
                elif rho(r, l, b) < rho_min:
                    #print("rho: ", rho(r, l, b))
                    continue # we are inside the Galaxy, but not within the bright H 2 regions, so we can skip this point
                else:
                    #print(integrand_axisymmetric(r, l, b) * dr * db)
                    value += integrand_axisymmetric(r, l, b) * dr * db
        #print(r, l, b, value*db)
        integral.append(value) # multiply by db to get the integral over the latitude
    # when done with integrating over r and b, return the value as we are not integrating over l
    integral = np.array(integral) / np.radians(1)  #np.radians(1) #(len(latitudes) * (np.radians(1))) #* np.radians(5)) # devide by delta-b and delta-l in radians, respectively, for the averaging the paper mentions
    print("len latitudes: ", len(latitudes))
    """ window_size = 5 / dl # 5 degrees in divided by the increment in degrees for the longitude. This is the window size for the running average, number of points
    integral = running_average(integral, window_size) / window_size #divide by delta-l in radians for the averaging the paper mentions """
    return longitudes, integral


def calc_luminocity():
    dr = 0.01   # increments in dr (kpc):
    dl = 0.1   # increments in dl (degrees):
    db = 0.1   # increments in db (degrees):

    # latitude range, integrate from -3.5 to 3.5 degrees, converted to radians
    latitudes = np.radians(np.arange(-3.5, 3.5 + db, db))

    # np.array with values for galactic longitude l in radians.
    l1 = np.arange(180, 0, -dl)
    l2 = np.arange(360, 180, -dl)
    longitudes = np.radians(np.concatenate((l1, l2)))

    # np.array with values for distance from the Sun to the star/ a point in the Galaxy
    radial_distances = np.arange(dr, r_s + rho_max + dr, dr) #r_s + rho_max is the maximum distance from the Sun to the outer edge of the Galaxy
    luminocity = 0
    for l in longitudes:
        for b in latitudes:
            for r in radial_distances:
                if rho(r, l, b) > rho_max:
                    break # we are outside of the Galaxy and are thus done integrating along this line of sight
                elif rho(r, l, b) < rho_min:
                    continue # we are inside the Galaxy, but not within the bright H 2 regions, so we can skip this point
                else:
                    luminocity += modelled_emissivity_axisymmetric(r, l, b)  * np.cos(b) * r**2 * dr * db
    return luminocity


def plot_axisymmetric():
    longitudes, integrated_spectrum = integral_axisymmetric()
    print(len(longitudes), len(integrated_spectrum))
    
    plt.plot(np.linspace(0, 100, len(longitudes)), integrated_spectrum)
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel("Galactic longitude l (degrees)")
    plt.ylabel("Modelled emissivity")
    plt.title("Modelled emissivity of the Galactic disk")
    plt.text(0.02, 0.95, fr'$H_\rho$ = {h} kpc & $\sigma_z$ = {sigma} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.9, fr'NII Luminosity = {integrated_spectrum:.2e} erg/s', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.text(0.02, 0.85, fr'{rho_min:.2e}  $\leq \rho \leq$ {rho_max:.2e} kpc', transform=plt.gca().transAxes, fontsize=8, color='black')
    plt.savefig("output/modelled_emissivity_axisymmetric_new33.png")  # save plot in the output folder
    #plt.show()

def plot_skymap():
    skydata = np.loadtxt('output/long_lat_skymap_axisymmetric.txt')
    print(skydata.shape)
    # Create coordinate grids
    long_grid, lat_grid = np.meshgrid(np.linspace(0, 100, len(skydata[1:, 0])), np.degrees(skydata[0, 1:]), indexing='ij')
    plt.pcolormesh(long_grid, lat_grid, skydata[1:, 1:], shading='auto')  
    plt.colorbar()
    # Redefine the x-axis labels to match the values in longitudes
    x_ticks = (180, 150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180)
    plt.xticks(np.linspace(0, 100, 13), x_ticks)
    plt.xlabel('Galactic Longitude (degrees)')
    plt.ylabel('Galactic Latitude (degrees)')
    plt.title('Skymap of the modelled luminocity')
    plt.savefig('output/skymap_axisymmetric.png', dpi=1200)
    plt.show()

# 15 with db = 0.01, dl = 0.01, dr = 0.01
# 16 with db = 0.1, dl = 0.001, dr = 0.01
# 17 with db = 0.1, dl = 0.01, dr = 0.01
# 18 with same as 17, with H_\rho = 3.34 (McKee & Williams, 1997)
# 19 with same as 17, with H_\rho = 2.4
# 20 with same as 17, \rho_min = 0, H_\rho = 2.5
# 21 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5
# 22 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 3.5 degrees, rho_min = 0
# 23 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 0.5 degrees, rho_min = 0, rho_max = 3
# 24 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 0.5 degrees, rho_min = 3, rho_max = 10, changed from break to continue. Did not make a difference
# 25 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 3.5 degrees, rho_min = 3, rho_max = 10 ############################################################################### BEST ONE SO FAR
# 26 with db = 0.5, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 3.5 degrees, rho_min = 3, rho_max = 10, do not divide by db, running average
# 27 with db = 0.5, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 3.5 degrees, rho_min = 3, rho_max = 10, do divide by db 
# 28 with db = 0.5, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 3.5 degrees, rho_min = 3, rho_max = 10, do divide by db, running average
# 29 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 0.5 degrees, rho_min = 3, rho_max = 10, do divide by db, running average
# 30 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 0.5 degrees, rho_min = 3, rho_max = 10, do not divide by db, running average
# 31 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 0.5 degrees, rho_min = 3, rho_max = 10, do not divide by db, no running average
# 32 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 0.5 degrees, rho_min = 3, rho_max = 10, do divide by db, no running average
# 33 with db = 0.1, dl = 0.1, dr = 0.01, H_rho = 2.5, latitude integration +- 3.5 degrees, rho_min = 3, rho_max = 10, do not divide by db, no running average


def plot_galacic_centric_distribution():
    r = np.linspace(rho_min, rho_max, 1000) # kpc, distance from the Galactic center
    plt.plot(r, axisymmetric_disk_model(r, h))
    plt.xlabel("Distance from the Galactic center (kpc)")
    plt.ylabel("Density")
    plt.title("Axisymmetric disk model with scale length h = " + str(h) + " kpc")
    plt.savefig("output/axisymmetric_disk_model.png")     # save plot in the output folder
    plt.show()


def main():
    """ start_time_old = time.time()
    integral_axisymmetric_old()
    end_time_old = time.time()
    runtime_old = end_time_old - start_time_old
    print("runtime old: ", runtime_old)

    start_time_new = time.time()
    integral_axisymmetric()
    end_time_new = time.time()
    runtime_new = end_time_new - start_time_new
    print("runtime new: ", runtime_new) """
    plot_skymap()
    #plot_axisymmetric()
    #luminocity = calc_luminocity()
    #print(luminocity)
    #lum = calc_luminocity()
    #print(lum)


if __name__ == "__main__":
    main()