import numpy as np
import src.observational_data.firas_data as firas_data
import src.utilities.constants as const
import src.utilities.settings as settings



def chi_squared(observational_data, modelled_data):
    # Ensure that we have no zeros in the modelled data set - set any zero equal to the minimal value in the array as to avoid dividing by zero
    modelled_data[modelled_data == 0] = np.min(modelled_data[modelled_data > 0])
    chi_squared = np.sum(((observational_data - modelled_data) ** 2) / modelled_data)
    return chi_squared


def load_data(filename_arm_intensities='intensities_per_arm_b_max_5.npy'):
    """ Function to load all components of the total NII intensity
    """
    bin_edges_line_flux, bin_centre_line_flux, line_flux, line_flux_error = firas_data.firas_data_for_plotting()
    intensities_per_arm = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/{filename_arm_intensities}')
    longitudes = np.lib.format.open_memmap(f'{const.FOLDER_GALAXY_DATA}/longitudes.npy')
    intensities_modelled = np.sum(intensities_per_arm[:4], axis=0)
    if settings.add_local_arm_to_intensity_plot == True: # add the local arm contribution
        intensities_modelled += intensities_per_arm[4]
    if settings.add_devoid_region_sagittarius == True: # take into account the known devoid region of Sagittarius
        intensities_modelled += intensities_per_arm[5]
    if settings.add_gum_cygnus == True: # add the contribution from the nearby OBA
        gum = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_gum.npy')
        cygnus = np.load(f'{const.FOLDER_GALAXY_DATA}/intensities_cygnus.npy')
        gum_cygnus = gum + cygnus
        intensities_modelled += gum_cygnus