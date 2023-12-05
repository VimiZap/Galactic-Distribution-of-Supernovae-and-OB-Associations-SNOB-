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

    def __init__(self, c, simulation_run_time):
        self.__n = int(np.ceil(np.exp((c - rng.random())/0.11))) # c = number of star formation episodes, n = number of SNPs in the association
        self.__simulation_run_time = simulation_run_time
        self.__supernovae = [] # list containting all the supernovae progenitors in the association
        self.__longitudes = []
        self.__exploded_sn = []

        self._calculate_position()
        self._generate_sn()
        self._find_longitudes()
        self._find_exploded_sn()
    
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
    def exploded_sn(self):
        return self.__exploded_sn
    
    @property
    def longitudes(self):
        return np.array(self.__longitudes)
    
    def _calculate_position(self):
        long_index = np.random.choice(a=len(self.densities_longitudinal), size=1, p=self.densities_longitudinal )
        lat_index = np.random.choice(a=len(self.densities_lat[long_index].ravel()), size=1, p=self.densities_lat[long_index].ravel() )
        radial_index = np.random.choice(a=len(self.rad_densities[:,long_index,lat_index].ravel()), size=1, p=self.rad_densities[:, long_index, lat_index].ravel() )
        grid_index = radial_index * 1800 * 21 + long_index * 21 + lat_index # 1800 = length of longitudes, 21 = length of latitudes
        self.__x = self.x_grid[grid_index]
        self.__y = self.y_grid[grid_index]
        self.__z = self.z_grid[grid_index]

    def _generate_sn(self):
        for i in range(self.__n):
            self.__supernovae.append(sn.SuperNovae(self.x, self.y, self.z, self.__simulation_run_time))
    
    def _find_longitudes(self):
        for sn in self.__supernovae:
            self.__longitudes.append(sn.long)

    def _find_exploded_sn(self):
        for sn in self.__supernovae:
            if sn.exploded:
                self.__exploded_sn.append(sn)

    @np.vectorize
    def vectorized_find_mass(sn):
        return sn.mass
    
    @property
    def find_sn_masses(self):
        return self.vectorized_find_mass(self.__supernovae)



if __name__ == '__main__':
    def test_association_distribution():
        a = 1
    test_ass = Association(1, 10)
    print(f"Test association contains {test_ass.number_sn} Supernovae Progenitors")
    print("The array of masses for the SN's:")
    print(test_ass.find_sn_masses)

