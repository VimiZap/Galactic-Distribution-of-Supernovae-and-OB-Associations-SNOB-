import numpy as np
import matplotlib.pyplot as plt

h = 2.5 # kpc, scale length of the disk
r_s = 8 # kpc, estimate for distance from the Sun to the Galactic center
r_min = 0.39 * r_s # kpc, minimum distance from the Sun to the Galactic center. Evaluates to 3.12 kpc
r_max = 1.30 * r_s # kpc, maximum distance from the Sun to the Galactic center. Evaluates to 10.4 kpc

def axisymmetric_disk_model(r, h): # r is the distance from the Galactic center
    return np.exp(-r/h)

def plot():
    r = np.linspace(r_min, r_max, 1000) # kpc, distance from the Galactic center
    plt.plot(r, axisymmetric_disk_model(r, h))
    plt.xlabel("Distance from the Galactic center (kpc)")
    plt.ylabel("Density")
    plt.title("Axisymmetric disk model with scale length h = " + str(h) + " kpc")
    plt.savefig("output/axisymmetric_disk_model.png")     # save plot in the output folder
    plt.show()


def axisymmetric_disk_model_2(N, r_s, h):
    """
    N: number of stars
    r_s: distance from the Sun to the Galactic center
    h: scale-length of the disk
    """
    r = np.random.exponential(h, N) # random numbers from the exponential distribution
    theta = np.random.uniform(0, 2*np.pi, N) # random numbers from the uniform distribution
    x = r*np.cos(theta) + r_s # x coordinates of the stars
    y = r*np.sin(theta) # y coordinates of the stars
    return x, y


def main():
    plot()

if __name__ == "__main__":
    main()