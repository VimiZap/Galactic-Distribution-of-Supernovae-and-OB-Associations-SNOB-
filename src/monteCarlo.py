import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes



def monteCarlo(n, C):
    N = [] # array with the number of SNP in each cluster
    for _ in range(n):
        r = rng.random() # random number between 0 and 1. 1 is excluded
        N.append(int(np.ceil(np.exp((C-r)/0.11))))
    return N

def plot(temporal_associations, star_formation_episodes):
    for i in range(len(temporal_associations)): 
        associations = temporal_associations[i]
        n = len(associations) # n = number of associations. Each element in 
        num_bins = int(np.ceil(max(associations))) # minimum number of stars = 1
        counts, _ = np.histogram(associations, bins=range(1, num_bins, 1))
        cumulative = (n - np.cumsum(counts))/n # cumulative distribution, normalized
        print("Average number of SN's in a cluster: " + str(np.average(associations)))
        plt.plot(range(1, num_bins-1, 1), cumulative, label= fr"{star_formation_episodes[temporal_associations.index(associations)]} episodes. Avg. number of SN's: {np.average(associations):.2f}")
    
    plt.xscale("log")
    plt.xlim(1, num_bins + 3000) # set the x axis limits
    plt.ylim(0, 1) # set the y axis limits
    plt.xlabel("Number of SNPs")
    plt.ylabel("Cumulative distribution. P(N > x)")
    plt.title("Monte Carlo simulation of temporal clustering of SNPs")
    plt.legend()
    plt.savefig("output/temporal_clustering.png")     # save plot in the output folder
    plt.show()


def clustering_MC():
    temporal_clustering = []
    star_formation_episodes = [1, 3, 5]
    for i in range(len(star_formation_episodes)):
        temporal_clustering.append(monteCarlo(1000000, C[i])) # 100000 is the number of clusters, and monteCarlo returns an array with the number of SNPs in each cluster
    plot(temporal_clustering, star_formation_episodes)


def main():
    clustering_MC()

if __name__ == "__main__":
    main()
