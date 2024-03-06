import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, sigma):
    return np.exp(-0.5 * x**2 / sigma**2) / (np.sqrt(2*np.pi) * sigma)

def exponential(x, tau):
    return np.exp(-x/tau)


x = np.linspace(0, 2, 1000)
sigma = 0.15
tau = 0.15

plt.plot(x, gaussian(x, sigma), label='Gaussian')
plt.plot(x, exponential(x, tau), label='Exponential')
plt.axvline(x = sigma, color = 'black', linestyle="--", label = 'axvline - full height')
plt.legend()
plt.show()
plt.close()