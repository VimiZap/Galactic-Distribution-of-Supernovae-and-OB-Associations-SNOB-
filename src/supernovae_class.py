import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

class SuperNovae:
    solar_masses = np.arange(8, 120, 0.01) # mass in solar masses
    m = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
    alpha = np.array([0.3, 1.3, 2.3, 2.7])
    seconds_in_myr = 3.156e13
    km_in_kpc = 3.2408e-17
    r_s = 8.178               # kpc, estimate for distance from the Sun to the Galactic center. Same value the atuhors used

    def __init__(self, association_x, association_y, association_z, simulation_run_time):
        """
        association_x, association_y, association_z: the position of the association in the galaxy in units of kpc
        simulation_run_time: the time of the simulation in units of Myr
        """
        self.__simulation_run_time = simulation_run_time
        self.__sn_mass = self._calculate_mass()
        self.__one_dim_vel = rng.normal(loc=0, scale=2, size=1) # Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
        self.__age = self._calculate_age() # Myr. Is either the age of the star or the time of death of the star, depending ont he simulation run time
        self.__exploded = False # True if the star has exploded, False otherwise

        # Calculate the position of the supernova
        self._calculate_position(association_x, association_y, association_z)

    # Other functions for calculating age, distance, etc.

    # Getter for velocity
    @property
    def velocity(self):
        return self.__one_dim_vel

    @property
    def age(self):
        return self.__age
    
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
    def long(self):
        return self.__long
    
    @property
    def exploded(self):
        return self.__exploded
    
    def _calculate_mass(self):
        imf = (self.m[2]/self.m[1])**(-self.alpha[1]) * (self.m[3]/self.m[2])**(-self.alpha[2]) * (self.solar_masses/self.m[3])**(-self.alpha[3])
        return np.random.choice(self.solar_masses, size=1, p=imf/np.sum(imf))
    
    def _calculate_age(self):
        # smallest SN progenitor mass is 8 solar masses, largest 120 solar masses. Ages range from 40 to 3 Myr. For now, assume linear relationship between mass and age.
        time_of_death = 40 + (self.mass - 8) * (3 - 40) / (120 - 8)
        if time_of_death > self.__simulation_run_time: # the star dies in the far future
            return self.__simulation_run_time # units of Myr
        else:
            self.__exploded = True
            return time_of_death # the star dies sometime before the end of simulation. Units of Myr
    
    def _calculate_position(self, association_x, association_y, association_z):
        # calculates the position of the supernova
        # draw random values for theta and phi
        theta = np.radians(rng.uniform(0, 2 * np.pi))
        phi = np.radians(rng.uniform(0, np.pi))
        r = self.velocity * self.seconds_in_myr * self.km_in_kpc * self.age
        self.__sn_x = r * np.sin(phi) * np.cos(theta) + association_x
        self.__sn_y = r * np.sin(phi) * np.sin(theta) + association_y
        self.__sn_z = r * np.cos(phi) + association_z
        self.__long = (np.arctan2(self.y - self.r_s, self.x) + np.pi/2) % (2 * np.pi)


    def plot(self, ax):
        # plots the supernova
        ax.scatter(self.x, self.y, self.z, c = 'b', s=1)

if __name__ == '__main__':    
    gum = SuperNovae(0, 0, 0, 5)
    cyg = SuperNovae(0, 0, 0, 5)
    pum = SuperNovae(0, 0, 0, 5)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    gum.plot(ax)
    cyg.plot(ax)
    pum.plot(ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    print(gum.velocity)
    print(gum.age)