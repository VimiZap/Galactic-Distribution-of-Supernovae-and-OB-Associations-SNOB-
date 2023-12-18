import numpy as np
import matplotlib.pyplot as plt
import supernovae_class as sn
rng = np.random.default_rng()

class Association():
    #galactic_densities = np.loadtxt('output\long_lat_skymap.txt')
    # positions:
    x_grid = np.load('output/galaxy_data/x_grid.npy')
    y_grid = np.load('output/galaxy_data/y_grid.npy')
    z_grid = np.load('output/galaxy_data/z_grid.npy')
    # densities:
    densities_longitudinal = np.load('output\galaxy_data\densities_longitudinal.npy')
    densities_longitudinal = densities_longitudinal/np.sum(densities_longitudinal) # normalize to unity
    densities_lat = np.load('output\galaxy_data\densities_lat.npy')
    densities_lat = densities_lat/np.sum(densities_lat, axis=1, keepdims=True) # normalize to unity for each latitude
    rad_densities = np.load('output\galaxy_data\densities_rad.npy')
    rad_densities = rad_densities/np.sum(rad_densities, axis=0, keepdims=True) # normalize to unity for each radius

    def __init__(self, c, creation_time, n=None):
        self.__n = self._calculate_num_sn(c, n)
        self.__creation_time = creation_time
        self.__simulation_time = creation_time # when the association is created, the simulation time is the same as the creation time
        self.__supernovae = [] # list containting all the supernovae progenitors in the association
        #self.__longitudes = []
        #self.__exploded_sn = []

        self._calculate_association_position()
        self._generate_sn()
    
    @property
    def number_sn(self):
        return self.__n
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def z(self):
        return self.__z
    
    @property
    def supernovae(self):
        return self.__supernovae
    
    """ @property
    def exploded_sn(self):
        return self.__exploded_sn
    
    @property
    def longitudes(self):
        return np.array(self.__longitudes)

    @np.vectorize
    def vectorized_find_mass(sn):
        return sn.mass
    
    @property
    def find_sn_masses(self):
        return self.vectorized_find_mass(self.__supernovae)
    
    @np.vectorize
    def vectorized_find_age(sn):
        return sn.age
    
    def find_sn_ages(self):
        return self.vectorized_find_age(self.__supernovae) """
    

    def _calculate_num_sn(self, c, n):
        if n==None:
            return int(np.ceil(np.exp((c - rng.random())/0.11))) # c = number of star formation episodes, n = number of SNPs in the association
        else:
            return n
    
    def _calculate_association_position(self):
        long_index = np.random.choice(a=len(self.densities_longitudinal), size=1, p=self.densities_longitudinal )
        lat_index = np.random.choice(a=len(self.densities_lat[long_index].ravel()), size=1, p=self.densities_lat[long_index].ravel() )
        radial_index = np.random.choice(a=len(self.rad_densities[:,long_index,lat_index].ravel()), size=1, p=self.rad_densities[:, long_index, lat_index].ravel() )
        grid_index = radial_index * 1800 * 21 + long_index * 21 + lat_index # 1800 = length of longitudes, 21 = length of latitudes
        self.__x = self.x_grid[grid_index]
        self.__y = self.y_grid[grid_index]
        self.__z = self.z_grid[grid_index]

    def _generate_sn(self):
        for i in range(self.__n):
            self.__supernovae.append(sn.SuperNovae(self.x, self.y, self.z, self.__creation_time, self.__simulation_time))

    def update_sn(self, new_simulation_time):
        #print(f"Updating SN's in association. New simulation time: {new_simulation_time} yrs.")
        for sn in self.__supernovae:
            sn.simulation_time = new_simulation_time
    
    def plot_association(self, ax, color='black'):
        ax.scatter(0, 0, 0, s=10, color='blue', label='Centre of association')
        """ ax.scatter(0, 0, 0, s=1, color='red', label='Exploded star', zorder=-1)
        ax.scatter(0, 0, 0, s=1, color='black', label='SN progenitor', zorder=-1) """
        
        for sn in self.__supernovae:
            sn.calculate_position()
            sn.plot_sn(ax, self.x, self.y, self.z, color)

    def print_association(self):
        print(f"Association contains {self.__n} Supernovae Progenitors and its centre is located at xyz position ({self.x}, {self.y}, {self.z}). Simulation time: {self.__simulation_time} yrs.")
        for sn in self.__supernovae:
            sn.print_sn()
        

    



if __name__ == '__main__':
    print("Running association_class.py")
    def test_association_distribution():
        a = 1
    test_ass = Association(1, 5)
    print(f"Test association contains {test_ass.number_sn} Supernovae Progenitors")
    print("The array of masses for the SN's:")
    print(test_ass.find_sn_masses)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    test_ass.vectorized_plot_sn(ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    test_ass.update_sn()



