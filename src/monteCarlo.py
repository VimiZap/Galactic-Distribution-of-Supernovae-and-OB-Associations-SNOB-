import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
C = [0.828, 0.95, 1.0] # values for the constant C for respectively 1, 3 and 5 star formation episodes



def monteCarlo(n, C):
    N = [] # number of stars formed
    for i in range(n):
        r = rng.random() # random number between 0 and 1. 1 is excluded
        N.append(np.exp((C - r)/0.11))
    return N

def plo2(data):
    # Sort the data in ascending order
    sorted_data = np.sort(data)

    # Calculate the cumulative sum of the sorted data
    cumulative_sum = np.cumsum(sorted_data)

    # Normalize the cumulative sum to get the cumulative distribution
    cumulative_distribution = cumulative_sum / cumulative_sum[-1]

    # Plot the cumulative distribution
    plt.plot(sorted_data, 1 - cumulative_distribution)
    
    # Make the x-axis logarithmic
    #plt.xscale("log")
    #plt.yscale("log")
    #plt.xlim(1, 10000) # set the x axis limits
    plt.show()


def main():
    N = monteCarlo(100000, C[0])
    plo2(N)

if __name__ == "__main__":
    main()
