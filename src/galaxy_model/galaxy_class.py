import logging
logging.basicConfig(level=logging.INFO) 
import numpy as np
import src.galaxy_model.association_class as asc
import src.utilities.utilities as ut
import src.galaxy_model.galaxy_density_distr as gdd

AVG_SN_PER_ASC = np.array([204, 620, 980]) # number of star formation episodes = 1, 3, 5
SN_BIRTHRATE = 2.81e4 # units of SN/Myr
ASC_BIRTHRATE = np.round(SN_BIRTHRATE / AVG_SN_PER_ASC).astype(int)  # number of associations created per Myr
STAR_FORMATION_EPISODES = [1, 3, 5]
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes

class Galaxy():
    rng = np.random.default_rng() # random number generator to be used for drawing association position
    # The following class variables are used to store the grid and the density distribution used for drawing the associations. 
    # If several Galaxy instances are created, these values are only generated once.
    x_grid = None
    y_grid = None
    z_grid = None
    uniform_spiral_arm_density = None # strictly not needed, but useful for testing
    emissivity = None
    
    
    def __init__(self, sim_time_duration, read_data_from_file = True, star_formation_episodes=1):
        """ Class to represent the galaxy. The galaxy is created at a given time and contains a number of associations.
        
        Args:
            sim_time_duration (int): The duration of the simulation in units of Myr
            read_data_from_file (bool, optional): Whether to read the Galaxy data from a file or generate them. Defaults to True (aka read from file).
            star_formation_episodes (int, optional): The number of star formation episodes. Defaults to 1.
            
        Returns:
            None
        """
        if not isinstance(sim_time_duration, int):
            raise TypeError("Simulation time duration must be an integer.")
        # Make sure that star_formation_episodes is a valid number of star formation episodes, i.e. that the number is in the list STAR_FORMATION_EPISODES
        if star_formation_episodes not in STAR_FORMATION_EPISODES:
            raise ValueError(f"Invalid number of star formation episodes. The number of star formation episodes must be one of the following: {STAR_FORMATION_EPISODES}")
        
        # Only generate data if it hasn't been generated yet
        if Galaxy.x_grid is None:
            self.generate_coords_densities(read_data_from_file)

        self._sim_time_duration = sim_time_duration # assuming the simulation time duration is in Myr
        self._star_formation_episodes = star_formation_episodes
        self._star_formation_episodes_index = STAR_FORMATION_EPISODES.index(star_formation_episodes)
        self._asc_birthrate = ASC_BIRTHRATE[self._star_formation_episodes_index]  # number of associations created per Myr
        self._galaxy = [] # List for storing all associations in the Galaxy
        self._generate_galaxy(sim_time_duration, self._asc_birthrate, C[self._star_formation_episodes_index])
    
    
    @classmethod
    def generate_coords_densities(cls, read_data_from_file):
        # generate the grid and the density distribution used for drawing the associations
        x_grid, y_grid, z_grid, uniform_spiral_arm_density, emissivity = gdd.generate_coords_densities(read_data_from_file)
        uniform_spiral_arm_density = uniform_spiral_arm_density / np.sum(uniform_spiral_arm_density) # normalize the density to unity
        emissivity = emissivity / np.sum(emissivity) # normalize the density to unity
        cls.x_grid = x_grid
        cls.y_grid = y_grid
        cls.z_grid = z_grid
        cls.uniform_spiral_arm_density = uniform_spiral_arm_density
        cls.emissivity = emissivity


    @ut.timing_decorator
    def _generate_galaxy(self, sim_time_duration, asc_birthrate, c):
        """ Method to generate the galaxy. The galaxy is created at a given time and contains a number of associations. Iterates over the simulation time and updates the associations and supernovae progenitors.
        
        Args:
            sim_time_duration (int): The duration of the simulation in units of Myr
            asc_birthrate (int): The number of associations created per Myr
            c (float): Number of star formation episodes
        
        Returns:
            None
        """
        self._calculate_association_position_batch(asc_birthrate, c, sim_time_duration) # add the first batch of associations, created at the beginning of the simulation
        logging.info(f'Simulation time: {sim_time_duration}')
        for sim_time in range(sim_time_duration - 1, -1, -1): # iterate over the simulation time, counting down to the present. sim_time_duration - 1 because the first batch of associations is already added
            if sim_time % 10 == 0:
                logging.info(f'Simulation time: {sim_time}')
            self._calculate_association_position_batch(asc_birthrate, c, sim_time)
        self._update_snps(0) # update the supernovae progenitors to the present time (0 Myr)
        self._update_exploded_supernovae() # update the list of exploded supernovae


    @property
    def galaxy(self):
        return self._galaxy
    
    @property
    def num_asc(self):
        return len(self._galaxy)
    
    @property
    def sim_time_duration(self):
        return self._sim_time_duration
    
    @property
    def asc_birthrate(self):
        return self._asc_birthrate
    
    @property
    def star_formation_episodes(self):
        return self._star_formation_episodes
    

    def _update_snps(self, sim_time):
        """ Method to update the supernovae progenitors to the given simulation time."""
        for association in self._galaxy:
            association.update_sn(sim_time)


    def _update_exploded_supernovae(self):
        """ Method to update the list of exploded supernovae. The list is updated at the end of the simulation."""
        exploded_sn = [] # list for storing all exploded supernovae
        for association in self._galaxy:
            for sn in association.supernovae:
                if sn.exploded:
                    exploded_sn.append(sn)
        self._exploded_sn = exploded_sn 


    def _calculate_association_position_batch(self, asc_birthrate, c, sim_time):
        """ Method to calculate the positions of the associations. The positions are calculated at each step of the simulation.
        
        Args:
            asc_birthrate (int): The number of associations created per Myr
            c (float): Number of star formation episodes
            sim_time (int): The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the creation_time to the present
            
        Returns:
            None
        """
        grid_indexes = self.rng.choice(a=len(self.emissivity), size=asc_birthrate, p=self.emissivity) 
        xs = self.x_grid[grid_indexes] # get the x-values for the drawn associations
        ys = self.y_grid[grid_indexes] # get the y-values for the drawn associations
        zs = self.z_grid[grid_indexes] # get the z-values for the drawn associations
        for i in range(asc_birthrate):
                self._galaxy.append(asc.Association(x=xs[i], y=ys[i], z=zs[i], association_creation_time=sim_time, c=c)) # add the association to the galaxy
    

    def get_exploded_supernovae_masses(self):
        """ Method to get the masses of the exploded supernovae progenitors."""
        exploded_sn_masses = [sn.mass for sn in self._exploded_sn]
        return exploded_sn_masses
    

    def get_exploded_supernovae_ages(self):
        """ Method to get the ages of the exploded supernovae progenitors."""
        exploded_sn_ages = [sn.age for sn in self._exploded_sn]
        return exploded_sn_ages
    
    
    def get_exploded_supernovae_longitudes(self):
        """ Method to get the longitudes of the exploded supernovae progenitors."""
        exploded_sn_longitudes = [sn.longitude for sn in self._exploded_sn]
        return exploded_sn_longitudes



def main():  
    Galaxy(1)
    Galaxy(50)

if __name__ == "__main__":
    main()