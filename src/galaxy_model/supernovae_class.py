import numpy as np
import src.utilities.constants as const


# real life snp Progenitors live from 3 - 40 Myrs  
class Supernovae:    
    def __init__(self, association_x, association_y, association_z, association_creation_time, simulation_time, sn_mass, one_dim_vel, expected_lifetime, vel_theta_dir, vel_phi_dir):
        """ Class to represent a snp progenitor. The snp progenitors are created at the same time as the association is created.
        
        Args:
            association_x (float): x-coordinate of the association. Units of kpc
            association_y (float): y-coordinate of the association. Units of kpc
            association_z (float): z-coordinate of the association. Units of kpc
            creation_time (int): how many years ago the sn/association was created. Units of Myr
            simulation_time (int): The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the creation_time to the present
            sn_mass (float): mass of the snp progenitor. Units of solar masses
            one_dim_vel (float): Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
            expected_lifetime (float): how many years it will take for the star to explode. Units of Myr
            vel_theta_dir (float): theta direction for the velocity dispersion. Units of radians
            vel_phi_dir (float): phi direction for the velocity dispersion. Units of radians

        Returns:
            None 
        """
        if (simulation_time > association_creation_time):
            raise ValueError("Simulation time can't be larger than snp creation time.")
        self.__association_x = association_x
        self.__association_y = association_y
        self.__association_z = association_z
        self.__sn_x = association_x # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
        self.__sn_y = association_y # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
        self.__sn_z = association_z # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
        self.__long = None # this is in radians! Value updated by calculate_position()
        self.__snp_creation_time = association_creation_time # how many years ago the sn/association was created
        self.__age = 0 # the age of the snp. Value updated by the setter
        self.__sn_mass = sn_mass
        self.__one_dim_vel = one_dim_vel # Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
        self.__expected_lifetime = expected_lifetime # Myr, how many years it will take for the star to explode
        self.__exploded = False # True if the star has exploded, False otherwise. Value dependent on creation time, age and expected_lifetime. 
        self.__vel_theta_dir = vel_theta_dir # radians
        self.__vel_phi_dir = vel_phi_dir # radians


    @property
    def age(self):
        return self.__age

    @property
    def expected_lifetime(self):
        return self.__expected_lifetime
    
    @property
    def snp_creation_time(self):
        return self.__snp_creation_time

    @property
    def velocity(self):
        return self.__one_dim_vel
    
    @property
    def mass(self):
        return self.__sn_mass
    
    @property
    def x(self): 
        return self.__sn_x
    
    @property
    def y(self): 
        return self.__sn_y
    
    @property
    def z(self): 
        return self.__sn_z
    
    @property
    def vel_theta_dir(self):
        return self.__vel_theta_dir
    
    @property
    def vel_phi_dir(self):
        return self.__vel_phi_dir
    
    @property
    def longitude(self):
        return self.__long
    
    @property
    def exploded(self):
        return self.__exploded
    

    def _calculate_age(self, sim_time):
        """ Function to calculate the age of the snp. The age is calculated at the given value for simulation_time.
        
        Args:
            sim_time (int): The new simulation time in units of Myr. Simulation time counts down to zero from the creation time of the Galaxy.
            
        Returns:
            None
        """
        if(sim_time > self.__snp_creation_time):
            raise ValueError("Simulation time can't be larger than snp creation time.")
        
        if not self.exploded: # only update the simulation time if the star has not exploded yet
            time_since_snp_creation = self.snp_creation_time - sim_time # how many years ago the snp was created
            if time_since_snp_creation >= self.expected_lifetime: # if True: the snp has exploded
                self.__age = self.expected_lifetime
                self.__exploded = True
            else: # if False: the snp has not exploded
                self.__age = time_since_snp_creation
                self.__exploded = False


    def _calculate_position(self):
        """ Function to calculate the position of the supernova. The position is calculated in Cartesian coordinates (x, y, z) (kpc) and the longitude (radians).
        Updates the private attributes __sn_x, __sn_y, __sn_z and __long."""
        r = self.__one_dim_vel * const.seconds_in_myr * const.km_in_kpc * self.age # radial distance travelled by the supernova in kpc
        self.__sn_x = r * np.sin(self.vel_theta_dir) * np.cos(self.vel_phi_dir) + self.__association_x # kpc
        self.__sn_y = r * np.sin(self.vel_theta_dir) * np.sin(self.vel_phi_dir) + self.__association_y # kpc
        self.__sn_z = r * np.cos(self.vel_theta_dir) + self.__association_z # kpc
        self.__long = (np.arctan2(self.y - const.r_s, self.x) + np.pi/2) % (2 * np.pi) # radians


    def update_snp(self, simulation_time):
        """ Method to update the snp. The age and position of the snp are updated for the given value for simulation_time.
        
        Args:
            simulation_time (int): The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the creation_time to the present
        
        Returns:
            None
        """
        self._calculate_age(simulation_time)
        self._calculate_position()


    def plot_sn(self, ax):
        """ Function to plot the SNP on an ax object, relative to the association centre. Positions are converted to pc for the plot.
        
        Args:
            ax (matplotlib.axes.Axes): The ax object on which to plot the supernova.
        
        Returns:
            None
        """

        if self.exploded:
            ax.scatter((self.x - self.__association_x) * 1e3, (self.y - self.__association_y) * 1e3, (self.z - self.__association_z) * 1e3, c='r', s=5)
        else:
            ax.scatter((self.x - self.__association_x) * 1e3, (self.y - self.__association_y) * 1e3, (self.z - self.__association_z) * 1e3, c='black', s=1)
    
    
    def print_sn(self):
        print(f"SNP is located at xyz position ({self.x}, {self.y}, {self.z}). Mass: {self.mass}, lifetime: {self.age} yrs, bool_exploded: {self.exploded}.")
