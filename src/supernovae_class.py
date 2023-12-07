import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

# real life SuperNovae lives from 3 - 40 Myrs
class SuperNovae:
    solar_masses = np.arange(8, 120, 0.01) # mass in solar masses
    m = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
    alpha = np.array([0.3, 1.3, 2.3, 2.7])
    seconds_in_myr = 3.156e13
    km_in_kpc = 3.2408e-17
    r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used
    # paramanters for the power law describing lifetime as function of mass. Schulreich et al. (2018)
    tau_0 = 1.6e8 * 1.65 #fits a little bit better with the data, though the slope is still too shallow
    beta = -0.932
    def __init__(self, association_x, association_y, association_z, creation_time, simulation_time):
        """
        association_x, association_y, association_z: the position of the association in the galaxy in units of kpc
        simulation_run_time: the time of the simulation in units of Myr
        """
        if (simulation_time > creation_time):
            raise ValueError("Simulation time can't be larger than supernovae creation time.")
        self.__association_x = association_x
        self.__association_y = association_y
        self.__association_z = association_z
        self.__sn_mass = self._calculate_mass()
        self.__one_dim_vel = rng.normal(loc=0, scale=2, size=1) # Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
        self.__creation_time = creation_time # how many years ago the sn/association was created
        self.__simulation_time = simulation_time # The current time of the simulation in units of Myr. In the simulation, this will decrease on each iteration, counting down from the past to the present
        self.__lifetime = self._calculate_lifetime() # Myr, how many years it will take for the star to explode
        self.__exploded = self._calculate_exploded() # True if the star has exploded, False otherwise. Value dependent on creation time, simulation time and lifetime. 
        self.__vel_theta_dir = rng.uniform(0, np.pi) # radians
        self.__vel_phi_dir = rng.uniform(0, 2 * np.pi) # radians

        # Calculate the position of the supernova
        self._calculate_position()

    # Other functions for calculating age, distance, etc.

    @property
    def simulation_time(self):
        return self.__simulation_time
    
    @simulation_time.setter # setter used to update the simulation run time. 
    def simulation_time(self, value):
        if(value > self.__creation_time):
            raise ValueError("Simulation time can't be larger than supernovae creation time.")
        
        if not self.exploded:
            self.__simulation_time = value # update the simulation run time
            self.__exploded = self._calculate_exploded() # update the age of the star
            self._calculate_position() # update the position of the star

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
    def long(self):
        return self.__long
    
    @property
    def exploded(self):
        return self.__exploded
    
    @property
    def age(self):
        if self.exploded:
            return self.lifetime
        else:
            return self.creation_time - self.simulation_time
    
    def _calculate_mass(self):
        imf = (self.m[2]/self.m[1])**(-self.alpha[1]) * (self.m[3]/self.m[2])**(-self.alpha[2]) * (self.solar_masses/self.m[3])**(-self.alpha[3])
        return np.random.choice(self.solar_masses, size=1, p=imf/np.sum(imf))
    
    def _calculate_lifetime(self):
        # smallest SN progenitor mass is 8 solar masses, largest 120 solar masses. Ages range from 40 to 3 Myr. For now, assume linear relationship between mass and age.
        # time_of_death = 40 + (self.mass - 8) * (3 - 40) / (120 - 8) # stupid simple linear relationship
        lifetime = self.tau_0 * (self.mass)**(self.beta) / 1e6 # devide to convert into units of Myr
        return lifetime
    
    def _calculate_exploded(self):
        # the star is born self.__creation_time Myr ago and has a lifetime of self.lifetime Myr.
        # If it has exploded, it did so (self.__creation_time - self.lifetime Myr ago). self.__simulation_time is x Myrs ago in the simulation.
        # I.e. far enough back into the past the star has not exploded yet, and as we evolve the system to the present, the star will explode at some point.
        return (self.__creation_time - self.lifetime) > self.__simulation_time
    
    def calculate_age(self):
        return 

    def _calculate_position(self):
        # calculates the position of the supernova
        # draw random values for theta and phi
        
        r = self.velocity * self.seconds_in_myr * self.km_in_kpc * (self.creation_time - self.simulation_time) # distance travelled by the supernova in kpc
        self.__sn_x = r * np.sin(self.vel_theta_dir) * np.cos(self.vel_phi_dir) + self.__association_x
        self.__sn_y = r * np.sin(self.vel_theta_dir) * np.sin(self.vel_phi_dir) + self.__association_y
        self.__sn_z = r * np.cos(self.vel_theta_dir) + self.__association_z
        self.__long = (np.arctan2(self.y - self.r_s, self.x) + np.pi/2) % (2 * np.pi)

    def plot_sn(self, ax, ass_x, ass_y, ass_z, color='black'):
        # plots the supernova relative to the association centre
        # multiply by 1e3 to convert from kpc to pc
        if self.exploded:
            ax.scatter((self.x - ass_x) * 1e3, (self.y - ass_y) * 1e3, (self.z - ass_z) * 1e3, c = 'r', s=5)
        else:
            ax.scatter((self.x - ass_x) * 1e3, (self.y - ass_y) * 1e3, (self.z - ass_z) * 1e3, c = color, s=1)
    
    def print_sn(self):
        print(f"Supernovae is located at xyz position ({self.x}, {self.y}, {self.z}). Mass: {self.mass}, lifetime: {self.age} yrs, bool_exploded: {self.exploded}.")

    

if __name__ == '__main__':    
    gum = SuperNovae(0, 0, 0, 50, 30)
    cyg = SuperNovae(0, 0, 0, 50, 30)
    pum = SuperNovae(0, 0, 0, 50, 30)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    gum.plot_sn(ax, 0, 0, 0)
    cyg.plot_sn(ax, 0, 0, 0)
    pum.plot_sn(ax, 0, 0, 0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    print("Velocities: ", gum.velocity, cyg.velocity, pum.velocity)
    print("Lifetimes: ", gum.lifetime, cyg.lifetime, pum.lifetime)
    print("Creation times: ", gum.creation_time, cyg.creation_time, pum.creation_time)
    print("Simulation times: ", gum.simulation_time, cyg.simulation_time, pum.simulation_time)
    print("Age of stars: ", gum.creation_time - gum.simulation_time, cyg.creation_time - cyg.simulation_time, pum.creation_time - pum.simulation_time)
    print("Masses: ", gum.mass, cyg.mass, pum.mass)