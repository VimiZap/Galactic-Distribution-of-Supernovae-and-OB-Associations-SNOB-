import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

rho_min = 2.9           # kpc, minimum distance from galactic center to the beginning of the spiral arms.
rho_max = 35 
pitch_angles = np.radians(np.array([13.5, 13.5, 13.5, 15.5]))
h = 2.5                 # kpc, scale length of the disk. The value Higdon and Lingenfelter used
arm_angles = np.radians([70, 160, 250, 340])

def spiral_arm_medians(arm_angle, pitch_angle):
    """
    Args:
        arm_angle: starting angle of the spiral arm, radians
        pitch_angle: pitch angle of the spiral arm, radians
    Returns:
        values for thetas and the corresponding rhos for the spiral arm
    """
    
    theta = [arm_angle]
    rho = [rho_min]
    dtheta = .01
    k = np.tan(pitch_angle)
    while rho[-1] < rho_max:
        theta.append((theta[-1] + dtheta) )#% (2*np.pi)) # the % is to make sure that theta stays between 0 and 2pi
        rho.append(rho_min*np.exp(k*(theta[-1] - theta[0])))
    
    print("Number of points for the given spiral arm: ", len(theta))
    return np.array(theta), np.array(rho)

def arm_median_density(rho): 
    """
    Args:
        rho: distance from the Galactic center
        h: scale length of the disk
    Returns:
        density of the disk at a distance rho from the Galactic center
    """
    return np.exp(-rho/h)

theta1, rho1 = spiral_arm_medians(arm_angles[0], pitch_angles[0])
x1, y1 = rho1*np.cos(theta1), rho1*np.sin(theta1)
theta2, rho2 = spiral_arm_medians(arm_angles[1], pitch_angles[1])
x2, y2 = rho2*np.cos(theta2), rho2*np.sin(theta2)
print(x2.shape, y2.shape)
print(len(x2[len(x2)//2 : -1]))
print(len(x2[0 : len(x2)//2]))
x2 = np.concatenate((x2[(len(x2))//2 - 1 : -1], x2[0 : (len(x2) )//2]))
y2 = np.concatenate((y2[(len(y2))//2 - 1 : -1], y2[0 : (len(y2) )//2]))

#print(x2)
#x2 = np.append(x2, x2[0 : len(x2)//2])
print(x2.shape, y2.shape)

densities1 = arm_median_density(rho1)
densities2 = griddata((x1, y1), densities1, (x2, y2), method='linear', fill_value=np.nan)
num_nan = np.count_nonzero(np.isnan(densities2))
#print(densities2[len(densities2)-145:-1])
print(num_nan)
#print(len(densities2))
plt.scatter(x1, y1, c=densities1, cmap='viridis', s=20, label='Spiral arm 1')
plt.scatter(x2, y2, c=densities2,  cmap='viridis', s=20, label='Spiral arm 2')
plt.legend()
# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Density')

# Set the aspect ratio to 'equal' for a more accurate representation of spherical coordinates
plt.gca().set_aspect('equal', adjustable='box')
plt.show()