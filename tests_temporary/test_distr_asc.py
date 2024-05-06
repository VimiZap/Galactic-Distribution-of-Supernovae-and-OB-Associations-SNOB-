import numpy as np
import matplotlib.pyplot as plt

def distr_1(N, slope):
    distr = 1/(N**(slope))
    print(np.sum(distr/   np.sum(distr)))
    return distr / np.sum(distr)



def distr_2(N):
    distr = 1/N**((1.6))
    print(np.sum(distr/   np.sum(distr)))
    
    return distr/   np.sum(distr)


def distr_3(N):
    distr = 1 - 0.11*np.log(N)
    print(np.sum(distr))
    print(np.sum(distr/   np.sum(distr)))
    return distr/   np.sum(distr)


def plot():
    N = np.arange(1, 1870)
    plt.plot(N, distr_1(N, slope=2), label='N^-2')
    plt.plot(N, distr_1(N, slope=1.6), label='N^-1.6')
    plt.plot(N, distr_1(N, slope=1.7), label='N^-1.7')
    #plt.plot(N, distr_3(N), label='0.828 - ln(N)')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(left=1, right=1870)
    plt.show()
    
    
plot()