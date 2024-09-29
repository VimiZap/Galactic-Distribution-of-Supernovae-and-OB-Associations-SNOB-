# Installation
To install the required packages, first set up a virtual environment on your machine and install the project as a package.

to create venv: 
- run: python3 -m venv venv (linux)
- run: python -m venv venv (windows)

- venv is activated if it says (.venv) in the terminal
- make sure the Python interpreter is selected for the .venv and not the global one

to activate venv: 
source venv/bin/activate  # Unix/Linux/macOS
venv\Scripts\activate     # Windows

to deactivate venv:
deactivate

to remove venv:
- deactive it first
- run: rm -rf .venv (linux)
- run: rmdir /s /q venv (windows)



To install project*:
- have a setup.py located in project root
- from project root, run: 'pip install -e .'


# To run the program
After installing the program as detailed above, the only parameter for the program you may want to configure is 'num_grid_subdivisions' located in src.utilities.settings.py, depending on the amount of RAM you have available. 
This parameter partitions the calculations into smaller parts, decreasing the amount of ram needed to run the program. The greater this value is, the more time is needed to finish the program. 
For a system with 32 GB ram we recommend setting this parameter to one, and for 8 GB ram to 4. 

The three booleans in this file determines wheter or not the code should include the local arm, the devoid region of Sagittarius-Carina and the Gum Nebula and Cygnus region to the intensity plot. 

In src.utilities.constants.py many different constants are located, including convertion factors, file paths, and values defining the model Galaxy. This includes for instance the Earth-GC distance (r_s) and parameters for the spiral arms.
Each constant in this file should be sufficiently explained, and the user can change these as they want to see how they affect the model.

The program is structured into several files. In the following, each file is briefly explained. The files which exist in the program but are not detailed here are not really meant to be run on their own but rather enters as componens of the other files.

In the folder src.nii_intensities:
- spiral_arm_model.py generates the data needed for the intensity plots for the spiral arm model. It also generates and saves these plots to file. 
- axisymmetric_disk_model.py generates the data needed for the intensity plot for the axisymmetric disk model. It also generates and saves this plot to file. 
- gum_cygnus.py is responsible for generating the data for the Gum nebula and the Cygnus region and saves this to file. If the user runs this script on its own, a intensity plot of these regions will be shown with the FIRAS data in the background
- analyze_density_height_dependence.py is more of a utility script, where the user can easier see the effect of the thickness of the Galaxy with different values for sigma
- chi_squared.py is used for optimizing the spiral arm parameters. The user controlls the number of iterations the script shall run with the parameter num_iterations which enters the function run_tests. It defaults to 10. The optimized values are printed to terminal

In the folder src.galaxy_model:
- combined_obas.py combines the modelled and known associations into one model for the Galactiv distribution of OBAs. These are used to make a plot of the Galaxy, and also generates some extra plots comparing the modelled and known associations. 

In the folder src.observational_data:
- analysis_obs_data.py generates statistics about the known associations. Data from this script enters the last three columns in Table 3 in the paper.
- firas_data.py contains functions related to the FIRAS data and data shared with us by Fixsen. Running this script generates two intensity plots, one for the FIRAS data and one for the data from Fixsen. 
  Also it calculates the intensity from the 30 degree bin in the FIRAS data, used for normalization of the axisymmetric model. Functions from this file are also used elsewhere in the program to add the FIRAS data to plots.
- rrl_hii.py simply plots the data of HII regions from the VizieR catalogue J/ApJS/165/338

In the folder tests:
- spiral_arm_model_tests.py contains functions to test the spiral arm model by varying the parameters of the model and plotting the resulting N II intensity.
- galaxy_test.py contains functions to test the Galaxy, Association and SNP classes. 

In the data folder:
- the fits file contains the data from FIRAS, retrieved from the web pages of NASA
- N+.txt contains the data shared by Fixsen
- Overview of known OB associations contains the data we collected from the litterature on the most prominent, known associations
- statistics_known_associations.csv is the result of running src.observational_data.analysis_obs_data.py