import numpy as np


# real life SuperNovae Progenitors live from 3 - 40 Myrs
class SuperNovae:
    seconds_in_myr = 3.156e13
    km_in_kpc = 3.2408e-17
    r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center
    
    def __init__(self, association_x, association_y, association_z, creation_time, simulation_time, sn_mass, one_dim_vel, lifetime, vel_theta_dir, vel_phi_dir):
        """ Class to represent a supernovae progenitor. The supernovae progenitors are created at the same time as the association is created.
        
        Args:
            association_x (float): x-coordinate of the association. Units of kpc
            association_y (float): y-coordinate of the association. Units of kpc
            association_z (float): z-coordinate of the association. Units of kpc
            creation_time (float): how many years ago the sn/association was created. Units of Myr
            simulation_time (float): The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the creation_time to the present
            sn_mass (float): mass of the supernovae progenitor. Units of solar masses
            one_dim_vel (float): Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
            lifetime (float): how many years it will take for the star to explode. Units of Myr
            vel_theta_dir (float): theta direction for the velocity dispersion. Units of radians
            vel_phi_dir (float): phi direction for the velocity dispersion. Units of radians

        Returns:
            None 
        """
        if (simulation_time > creation_time):
            raise ValueError("Simulation time can't be larger than supernovae creation time.")
        self.__association_x = association_x
        self.__association_y = association_y
        self.__association_z = association_z
        self.__sn_x = association_x # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
        self.__sn_y = association_y # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
        self.__sn_z = association_z # because the SNP's are born at the association centre. Also prevents issues if the getters for SNP positions are called before calculate_position()
        self.__long = None # this is in radians! Value updated by calculate_position()
        self.__sn_mass = sn_mass
        self.__one_dim_vel = one_dim_vel # Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
        self.__creation_time = creation_time # how many years ago the sn/association was created
        self.__simulation_time = simulation_time # The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the past to the present
        self.__lifetime = lifetime # Myr, how many years it will take for the star to explode
        self.__exploded = False # True if the star has exploded, False otherwise. Value dependent on creation time, simulation time and lifetime. 
        self.__vel_theta_dir = vel_theta_dir # radians
        self.__vel_phi_dir = vel_phi_dir # radians


    @property
    def simulation_time(self):
        return self.__simulation_time
    
    @simulation_time.setter # setter used to update the simulation run time. 
    def simulation_time(self, value):
        if(value > self.__creation_time):
            raise ValueError("Simulation time can't be larger than supernovae creation time.")
        
        if not self.exploded: # only update the simulation time if the star has not exploded yet
            self.__simulation_time = value # update the simulation run time
            self.__exploded = self._calculate_exploded() # update the age of the star

    @property
    def lifetime(self):
        return self.__lifetime
    
    @property
    def creation_time(self):
        return self.__creation_time

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
    
    @property
    def age(self):
        if self.exploded: # if the star has exploded, return the lifetime as given by the formula used from Schulreich et al. (2018) (as the star does not age anymore after it has exploded)
            return self.lifetime
        else:
            return self.creation_time - self.simulation_time 
    
    def _calculate_exploded(self):
        """ Function to calculate if the star has exploded. Returns True if the star has exploded, False otherwise.
        
        Args:
            None
        
        Returns:
            bool: True if the star has exploded, False otherwise
        """
        # the star is born self.__creation_time Myr ago and has a lifetime of self.lifetime Myr.
        # If it has exploded, it did so (self.__creation_time - self.lifetime Myr ago). self.__simulation_time is x Myrs ago in the simulation.
        # I.e. far enough back into the past the star has not exploded yet, and as we evolve the system to the present, the star will explode at some point.
        return (self.__creation_time - self.lifetime) > self.__simulation_time


    def calculate_position(self):
        """ Function to calculate the position of the supernova. The position is calculated in Cartesian coordinates (x, y, z) (kpc) and the longitude (radians).
        Updates the private attributes __sn_x, __sn_y, __sn_z and __long.

        Args:
            None
        
        Returns:
            None    
        """
        r = self.__one_dim_vel * self.seconds_in_myr * self.km_in_kpc * (self.creation_time - self.simulation_time) # radial distance travelled by the supernova in kpc
        self.__sn_x = r * np.sin(self.vel_theta_dir) * np.cos(self.vel_phi_dir) + self.__association_x # kpc
        self.__sn_y = r * np.sin(self.vel_theta_dir) * np.sin(self.vel_phi_dir) + self.__association_y # kpc
        self.__sn_z = r * np.cos(self.vel_theta_dir) + self.__association_z # kpc
        self.__long = (np.arctan2(self.y - self.r_s, self.x) + np.pi/2) % (2 * np.pi) # radians

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
        print(f"Supernovae is located at xyz position ({self.x}, {self.y}, {self.z}). Mass: {self.mass}, lifetime: {self.age} yrs, bool_exploded: {self.exploded}.")
