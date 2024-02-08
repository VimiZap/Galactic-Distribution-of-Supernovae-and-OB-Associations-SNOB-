import numpy as np
import matplotlib.pyplot as plt
import supernovae_class as sn
import utilities as ut
rng = np.random.default_rng()

class Association():
    #galactic_densities = np.loadtxt('output\long_lat_skymap.txt')
    ### Parameters for the IMF:
    tau_0 = 1.6e8 * 1.65 #fits a little bit better with the data, though the slope is still too shallow
    beta = -0.932
    alpha = np.array([0.3, 1.3, 2.3, 2.7])
    m = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
    solar_masses = np.arange(8, 120, 0.01) # mass in solar masses
    imf = (m[2]/m[1])**(-alpha[1]) * (m[3]/m[2])**(-alpha[2]) * (solar_masses/m[3])**(-alpha[3])
    # 
    num_lats = len(np.lib.format.open_memmap('output/galaxy_data/latitudes.npy'))
    num_rads = len(np.lib.format.open_memmap('output/galaxy_data/radial_distances.npy'))
    num_longs = len(np.lib.format.open_memmap('output/galaxy_data/longitudes.npy'))
    # positions:
    x_grid = np.lib.format.open_memmap('output/galaxy_data/x_grid.npy')
    y_grid = np.lib.format.open_memmap('output/galaxy_data/y_grid.npy')
    z_grid = np.lib.format.open_memmap('output/galaxy_data/z_grid.npy')
    # densities:
    emissivity_longitudinal = np.load('output/galaxy_data/emissivity_longitudinal.npy')
    emissivity_longitudinal = emissivity_longitudinal/np.sum(emissivity_longitudinal) # normalize to unity
    emissivity_lat = np.load('output/galaxy_data/emissivity_long_lat.npy')
    emissivity_lat = emissivity_lat/np.sum(emissivity_lat, axis=1, keepdims=True) # normalize to unity for each latitude
    emissivitty_rad = np.load('output/galaxy_data/emissivity_rad_long_lat.npy')
    emissivitty_rad = emissivitty_rad/np.sum(emissivitty_rad, axis=0, keepdims=True) # normalize to unity for each radius

    def __init__(self, c, creation_time, n=None):
        self.__n = self._calculate_num_sn(c, n)
        self.__creation_time = creation_time
        self.__simulation_time = creation_time # when the association is created, the simulation time is the same as the creation time
        self._calculate_association_position()
        self._generate_sn_batch() # list containting all the supernovae progenitors in the association

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
    
    def _calculate_num_sn(self, c, n):
        if n==None:
            return int(np.ceil(np.exp((c - rng.random())/0.11))) # c = number of star formation episodes, n = number of SNPs in the association
        else:
            return n
    
    def _calculate_association_position(self):
        long_index = np.random.choice(a=len(self.emissivity_longitudinal), size=1, p=self.emissivity_longitudinal )
        lat_index = np.random.choice(a=len(self.emissivity_lat[long_index].ravel()), size=1, p=self.emissivity_lat[long_index].ravel() )
        radial_index = np.random.choice(a=len(self.emissivitty_rad[:,long_index,lat_index].ravel()), size=1, p=self.emissivitty_rad[:, long_index, lat_index].ravel() )
        grid_index = radial_index * self.num_longs * self.num_lats + long_index * self.num_lats + lat_index # 1800 = length of longitudes, 21 = length of latitudes
        self.__x = self.x_grid[grid_index]
        self.__y = self.y_grid[grid_index]
        self.__z = self.z_grid[grid_index]

    def _generate_sn_batch(self):
        sn_masses = np.random.choice(self.solar_masses, size=self.__n, p=self.imf/np.sum(self.imf))
        one_dim_velocities = rng.normal(loc=0, scale=2, size=self.__n)
        lifetimes = self.tau_0 * (sn_masses)**(self.beta) / 1e6 # devide to convert into units of Myr
        vel_theta_dirs = rng.uniform(0, np.pi, size=self.__n)
        vel_phi_dirs = rng.uniform(0, 2 * np.pi, size=self.__n)
        self.__supernovae = [sn.SuperNovae(self.x, self.y, self.z, self.__creation_time, self.__simulation_time, sn_masses[i], one_dim_velocities[i], lifetimes[i], vel_theta_dirs[i], vel_phi_dirs[i]) for i in range(self.__n)]
    
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



