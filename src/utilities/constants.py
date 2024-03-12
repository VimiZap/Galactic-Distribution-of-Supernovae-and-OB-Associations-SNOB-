import numpy as np

# constants for the axisymmetric and spiral arm models
h_spiral_arm = 2.4    # kpc, scale length of the disk. The value Higdon and Lingenfelter used
h_axisymmetric = 2.5  # kpc, scale length of the disk. The value Higdon and Lingenfelter used
r_s = 8.178            # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
# This rho_max and rho_min seems to be taken from Valee
rho_min_spiral_arm = 2.9           # kpc, minimum distance from galactic center to the beginning of the spiral arms. Values taken from Valleé (see Higdon and lingenfelter)
rho_max_spiral_arm = 35            # kpc, maximum distance from galactic center to the end of the spiral arms. Values taken from Valleé (see Higdon and lingenfelter)
rho_min_axisymmetric = 3           # kpc, minimum distance from galactic center to bright H 2 regions. Evaluates to 3.12 kpc
rho_max_axisymmetric = 11          # kpc, maximum distance from galactic center to bright H 2 regions. Evaluates to 10.4 kpc
sigma_height_distr = 0.15          # kpc, scale height of the disk
sigma_arm = 0.5                    # kpc, dispersion of the spiral arms
total_galactic_n_luminosity = 1.4e40    #total galactic N 2 luminosity in erg/s
gum_nii_luminosity = 1e36 # erg/s, luminosity of the Gum Nebula in N II 205 micron line. Number from Higdon and Lingenfelter
cygnus_nii_luminosity = 2.4e37 # erg/s, luminosity of the Cygnus Loop in N II 205 micron line. Number from Higdon and Lingenfelter
measured_nii_30_deg = 0.00011711056373558678 # erg/s/cm²/sr measured N II 205 micron line intensity at 30 degrees longitude. Retrieved from the FIRAS data with the function firas_data.find_firas_intensity_at_central_long(30)
kpc = 3.08567758e21    # 1 kpc in cm
# kpc^2, source-weighted Galactic-disk area. See https://iopscience.iop.org/article/10.1086/303587/pdf, equation 37
a_d_spiral_arm = 2*np.pi*h_spiral_arm**2 * ((1+rho_min_spiral_arm/h_spiral_arm)*np.exp(-rho_min_spiral_arm/h_spiral_arm) - (1+rho_max_spiral_arm/h_spiral_arm)*np.exp(-rho_max_spiral_arm/h_spiral_arm))
a_d_axisymmetric = 2*np.pi*h_axisymmetric**2 * ((1+rho_min_axisymmetric/h_axisymmetric)*np.exp(-rho_min_axisymmetric/h_axisymmetric) - (1+rho_max_axisymmetric/h_axisymmetric)*np.exp(-rho_max_axisymmetric/h_axisymmetric)) 
# starting angles, pitch-angles and fractional contributions for the spiral arms, respectively Norma-Cygnus(NC), Perseus(P), Sagittarius-Carina(SA), Scutum-Crux(SC)
arm_angles = np.radians([65, 160, 240, 330])  # best fit for the new r_s
pitch_angles = np.radians([14, 14, 14, 16]) # best fir to new r_s
fractional_contribution = [0.18, 0.36, 0.18, 0.28] 
spiral_arm_names = ['Norma-Cygnus', 'Perseus', 'Sagittarius-Carina', 'Scutum-Crux']
number_of_end_points = 45 # number of points to use for the circular projection at the end points of the spiral arms


# Stuff for SN, Ass and Galaxy classes:
seconds_in_myr = 3.156e13
km_in_kpc = 3.2408e-17

# Parameters for the modified Kroupa IMF:
alpha = np.array([0.3, 1.3, 2.3, 2.7])
m_lim_imf_powerlaw = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
# paramanters for the power law describing lifetime as function of mass. Schulreich et al. (2018)
tau_0 = 1.6e8 * 1.65 # fits better with the data for he higher masses, though the slope is still too shallow
beta = -0.932

# Folder locations
FOLDER_GALAXY_DATA = 'galaxy_data'
FOLDER_OBSERVATIONAL_DATA = 'data/observational'
FOLDER_GALAXY_TESTS = 'data/plots/tests_galaxy_class'
FOLDER_MODELS_GALAXY = 'data/plots/models_galaxy'
FOLDER_OBSERVATIONAL_PLOTS = 'data/plots/observational_plots'