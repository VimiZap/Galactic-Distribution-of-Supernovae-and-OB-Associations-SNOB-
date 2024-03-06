import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import src.galaxy_model.supernovae_class as sn
import src.utilities.utilities as ut


class Association():
    # paramanters for the power law describing lifetime as function of mass. Schulreich et al. (2018)
    tau_0 = 1.6e8 * 1.65 # fits better with the data for he higher masses, though the slope is still too shallow
    beta = -0.932
    # Parameters for the IMF:
    alpha = np.array([0.3, 1.3, 2.3, 2.7])
    m = np.array([0.01, 0.08, 0.5, 1, 120]) # mass in solar masses. Denotes the limits between the different power laws
    solar_masses = np.arange(8, 120, 0.01) # mass in solar masses. Used to draw random masses for the SNPs in the association
    imf = (m[2]/m[1])**(-alpha[1]) * (m[3]/m[2])**(-alpha[2]) * (solar_masses/m[3])**(-alpha[3]) # the modified Kroupa initial mass function
    rng = np.random.default_rng()

    def __init__(self, c, creation_time, x, y, z, n=None):
        """ Class to represent an association of supernovae progenitors. The association is created at a given position and time. 
        The number of SNPs in the association is calculated from the number of star formation episodes and IMF. 
        The SNPs are generated at the time of the association's creation.
        
        Args:
            c: number of star formation episodes
            creation_time: how many years ago the association was created. Units of Myr
            x: x-coordinate of the association. Units of kpc
            y: y-coordinate of the association. Units of kpc
            z: z-coordinate of the association. Units of kpc
            n: number of SNPs in the association. If None, a random number of SNPs is drawn. Otherwise, the given number of SNPs is used.
            
        Returns:
            None
        """
        self.__x = x
        self.__y = y
        self.__z = z
        self.__creation_time = creation_time
        self.__simulation_time = creation_time # when the association is created, the simulation time is the same as the creation time. Simulation_time = creation_time - time passed since creation. Goes down to 0
        self.__n = self._calculate_num_sn(c, n)
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
        """ Function to calculate the number of SNPs in the association. If n is None, a random number of SNPs is drawn. Otherwise, the given number of SNPs is used.
        
        Args:
            c: number of star formation episodes
            n: number of SNPs in the association

        Returns:
            int: number of SNPs in the association
        """
        if n==None: # draw random number of SNPs
            return int(np.ceil(np.exp((c - self.rng.random())/0.11))) # c = number of star formation episodes, n = number of SNPs in the association
        else: # use the given number of SNPs
            return n
    

    def _generate_sn_batch(self):
        """ Function to generate a batch of SNPs. The number of SNPs is given by the attribute self.__n and stored in the list self.__supernovae.
        Each SNP is given a random mass, a random velocity, a random lifetime and a random direction for the velocity dispersion. 
        The random values are drawn from the initial mass function and a Gaussian distribution.
        
        Args:
            None

        Returns:
            None
        """
        sn_masses = self.rng.choice(self.solar_masses, size=self.__n, p=self.imf/np.sum(self.imf)) # draw random masses for the SNPs in the association
        one_dim_velocities = self.rng.normal(loc=0, scale=2, size=self.__n) # Gaussian velocity distribution with a mean of 0 km/s and a standard deviation of 2 km/s
        lifetimes = self.tau_0 * (sn_masses)**(self.beta) / 1e6 # Devide to convert into units of Myr. Formula from Schulreich et al. (2018)
        vel_theta_dirs = self.rng.uniform(0, np.pi, size=self.__n)   # Velocity dispersion shall be isotropic
        vel_phi_dirs = self.rng.uniform(0, 2 * np.pi, size=self.__n) # Velocity dispersion shall be isotropic
        self.__supernovae = [sn.SuperNovae(self.x, self.y, self.z, self.__creation_time, self.__simulation_time, sn_masses[i], one_dim_velocities[i], 
                                           lifetimes[i], vel_theta_dirs[i], vel_phi_dirs[i]) for i in range(self.__n)]
    

    def update_sn(self, new_simulation_time): # update each individual SNP in the association
        """ Function to update the simulation time for each SNP in the association. The simulation time is updated to the new_simulation_time, and the boolean exploded attribute is also updated.
        
        Args:
            new_simulation_time: the new simulation time in units of Myr
            
        Returns:
            None
        """
        for sn in self.__supernovae:
            sn.simulation_time = new_simulation_time # Calls the setter in the SuperNovae class to update the simulation time for each SNP in the association, as well as the boolean exploded attribute
    

    def plot_association(self, ax):
        """ Function to plot the association in the galactic plane, with the centre of the association as origo.
        The centre of the association is plotted in blue, and the SNPs are plotted in red if they have exploded, and black if they have not exploded.
        
        Args:
            ax: axis object from matplotlib to which the association is plotted
        
        Returns:
            None
        """
        ax.scatter(0, 0, 0, s=10, color='blue', label='Centre of association') # plot the centre of the association
        for sn in self.__supernovae: # plot the SNPs in the association. Both exploded (red) and unexploded (black)
            sn.calculate_position()
            sn.plot_sn(ax)


    def print_association(self, prin_snp=False):
        """ Function to print the association. It prints the number of SNPs in the association, the position of the centre of the association and the simulation time. If prin_snp is True, it also prints info on the SNPs in the association.
        
        Args:
            prin_snp: boolean, if True, prints info on the SNPs in the association
            
        Returns:
            None
        """
        print(f"Association contains {self.__n} Supernovae Progenitors and its centre is located at xyz position ({self.x}, {self.y}, {self.z}). Simulation time: {self.__simulation_time} yrs.")
        if prin_snp:
            for sn in self.__supernovae:
                sn.print_sn()
