import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np


def plot_age_hist(age_data_known, age_data_modelled, filename, bin_max_age: int = 50):
    """ Plot the age vs. distance of OB associations
    
    Args:
        age_data: array. Age of the associations
    
    Returns:
        None. Shows the plot
    """
    binwidth = 1
    bin_max_age = np.max(age_data_modelled)
    bins = np.arange(0, bin_max_age + binwidth, binwidth)
    plt.figure(figsize=(10, 6))
    plt.hist(age_data_known, bins=bins, label='Known associations', alpha=0.5)
    plt.hist(age_data_modelled, bins=bins, label='Modelled associations', alpha=0.5)
    #plt.title('Histogram of ages of OB associations')
    plt.xlabel('Age (Myr)', fontsize=12)
    plt.xlim(0, bin_max_age)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10)) 
    plt.ylabel('Counts', fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # Set the y-axis to only use integer ticks
    plt.grid(axis='y')
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    #plt.show()
    plt.savefig(filename)
    plt.close()


age_data_known = np.random.randint(0, 50, 100)
age_data_modelled = np.random.randint(0, 50, 100)

plot_age_hist(age_data_known, age_data_modelled, 'age_hist_font_tests.pdf')