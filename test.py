from scipy.integrate import quad
from scipy.integrate import dblquad
import numpy as np
def integrand(x, y, a, b):
    return a*x**2 + b*y

def dbleintegrand(y, x, a, b):
    return a*x**2 + b*y

a = 2
b = 2
y = np.linspace(0, 1, 100)


def funcint(y, a, b):
    return quad(integrand, 0, 1, args=(y,a,b))

def funcintvec(y, a, b):
    return np.vectorize(funcint)(y, a, b)

def funcdoubleint(a, b):
    return dblquad(dbleintegrand, 0, 1, 0, 1, args=(a,b))

dl = 0.5
a1 = np.arange(180, 0, -0.01)
l2 = np.arange(359, 179, -dl)
print(a1)
print(l2)
""" I = funcintvec(y, a, b)
i = funcint(0.5, a, b)
i2 = quad(integrand, 0, 1, args=(0.5,a,b))
print(I)
print(i)
print(i2)
doubleint = funcdoubleint(a, b)
print(doubleint) """