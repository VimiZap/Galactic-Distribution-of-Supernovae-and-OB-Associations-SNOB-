import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes



def monteCarlo(n, C):
    N = [] # array with the number of SNP in each cluster
    for _ in range(n):
        r = rng.random() # random number between 0 and 1. 1 is excluded
        N.append(np.exp((C - r)/0.11))
    return N

def plot(Data, star_formation_episodes):
    for data in Data: 
        n = len(data)
        num_bins = int(np.ceil(max(data))) # minimum number of stars = 0
        counts, _ = np.histogram(data, bins=range(0, num_bins, 1))
        cumulative = (n - np.cumsum(counts))/n # cumulative distribution, normalized
        plt.plot(range(1, num_bins, 1), cumulative, label="Number of star formation episodes = " + str(star_formation_episodes[Data.index(data)]))
    
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
        temporal_clustering.append(monteCarlo(100000, C[i]))
    plot(temporal_clustering, star_formation_episodes)


def main():
    clustering_MC()

if __name__ == "__main__":
    main()
