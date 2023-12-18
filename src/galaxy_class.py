import numpy as np
import matplotlib.pyplot as plt
import association_class as asc
import time as time

AVG_SN_PER_ASC = np.array([204, 620, 980]) # number of star formation episodes = 1, 3, 5
SN_BIRTHRATE = 2.81e4 # units of SN/Myr
STAR_FORMATION_EPISODES = [1, 3, 5]
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to run.")
        return result
    return wrapper

@np.vectorize
def vectorized_update_sn(associations, new_simulation_time): # syntax sugar for a for loop
    associations.update_sn(new_simulation_time)

@timing_decorator
class Galaxy():
    def __init__(self, sim_time_duration, star_formation_episodes=1):
        if not isinstance(sim_time_duration, int):
            raise TypeError("Simulation time duration must be an integer.")
        # assuming the simulation time duration is in Myr
        self._sim_time_duration = sim_time_duration
        self._star_formation_episodes = star_formation_episodes
        self._star_formation_episodes_index = STAR_FORMATION_EPISODES.index(self._star_formation_episodes)
        self._asc_birthrate = int(np.round(SN_BIRTHRATE / AVG_SN_PER_ASC[self._star_formation_episodes_index]))  # number of associations created per Myr
        #print(self._asc_birthrate)
        #t.sleep(2)
        self._galaxy = [] # empty list containing all the associations in the galaxy
    
        self._generate_galaxy(sim_time_duration, self._asc_birthrate, C[self._star_formation_episodes_index])
    
    
    def _generate_galaxy(self, sim_time_duration, asc_birthrate, c):
        for sime_time in range(sim_time_duration, -1, -1):
            print(sime_time)
            if sime_time != sim_time_duration:
                vectorized_update_sn(self._galaxy, sime_time)
            for num_asc in range(asc_birthrate):
                #print(num_asc)
                self._galaxy.append(asc.Association(c, sime_time))
        self._update_exploded_supernovae()
        self._calculate_sn_pos()
    
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
    
    def _update_exploded_supernovae(self):
        exploded_sn = []
        for association in self._galaxy:
            for sn in association.supernovae:
                if sn.exploded:
                    exploded_sn.append(sn)
        self._exploded_sn = exploded_sn
        
    def _calculate_sn_pos(self):
        for sn in self._exploded_sn:
            sn.calculate_position()
    
    def get_exploded_supernovae_masses(self):
        exploded_sn_masses = [sn.mass for sn in self._exploded_sn]
        return exploded_sn_masses
    
    def get_exploded_supernovae_ages(self):
        exploded_sn_ages = [sn.age for sn in self._exploded_sn]
        return exploded_sn_ages
    
    
    def get_exploded_supernovae_longitudes(self):
        exploded_sn_longitudes = [sn.longitude for sn in self._exploded_sn]
        return exploded_sn_longitudes



def main():  
    Galaxy(1)
    Galaxy(2)
    Galaxy(100)

if __name__ == "__main__":
    main()