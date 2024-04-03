import numpy as np
import matplotlib.pyplot as plt
#import src.utilities.constants as const
import src.utilities.utilities as util



""" def add_sag_to_ax(ax, x, y):
    ax.scatter(x, y, label='sc', s=3)
    ax.scatter(0, const.r_s, c='r', label='Sun')
    ax.scatter(0,0, c='y', label='Galactic center')
    ax.set_aspect('equal', adjustable='box') """
    

def add_los_to_ax(ax, x, y):
    ax.scatter(x, y, label='Line of sight', s=3)


def find_rho_min(ax, long, r_los_start):
    theta_sa, rho_sa = util.spiral_arm_medians(np.radians(240), np.radians(14), rho_max=8) # generate the spiral arm medians for sagittarius-carina
    x_sa = rho_sa*np.cos(theta_sa)
    y_sa = rho_sa*np.sin(theta_sa)
    dr = 0.1
    rs = np.arange(r_los_start, r_los_start + 7 + dr, dr)
    theta_los = util.theta(rs, np.radians(long), 0)
    rho_los = util.rho(rs, np.radians(long), 0)
    x_los = rho_los*np.cos(theta_los)
    y_los = rho_los*np.sin(theta_los)
    ax.scatter(x_los, y_los, label=f'Line of sight for {long}°', s=3)
    x_diff = (x_los[:, np.newaxis] - x_sa) ** 2
    y_diff = (y_los[:, np.newaxis] - y_sa) ** 2
    dists = np.sqrt(x_diff + y_diff) # distances between each point along the line of sight and every point in the spiral arm
    dist_min = np.min(dists)
    print(f'Minimum distance: {dist_min}')
    dist_min_index_rho = np.argwhere(dists == dist_min)[0][1] # index of the minimum distance in the rho_sa array
    print(dist_min_index_rho)
    rho_closest = rho_sa[dist_min_index_rho]
    theta_closest = theta_sa[dist_min_index_rho]
    x_closest = rho_closest*np.cos(theta_closest)
    y_closest = rho_closest*np.sin(theta_closest)
    return x_closest, y_closest, rho_closest
    
    

def find_rho_min_max(long1=30, long2=55):
    fig, ax = plt.subplots()
    x_1, y_1, rho_closest1 = find_rho_min(ax, long1, 0)
    x_2, y_2, rho_closest2 = find_rho_min(ax, long1, 7)
    distance = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    longer_distance = np.sqrt((5-x_1)**2 + (0.5-y_1)**2)
    print('Distance between the two closest points:', distance)
    print('Distance between closest point and intersection with its line of sight with x-axis:', longer_distance)
    print('Angle above the x-axis long 1: ', np.degrees(np.arctan2(y_1,x_1)))
    print('Angle above the x-axis long 2: ', np.degrees(np.arctan2(y_2,x_2)))
    
    #theta_sa, rho_sa = util.spiral_arm_medians(const.arm_angles[2], const.pitch_angles[2], rho_max=8) # generate the spiral arm medians for sagittarius-carina
    #x_sa = rho_sa*np.cos(theta_sa)
    #y_sa = rho_sa*np.sin(theta_sa)
    #ax.scatter(x_1, y_1, c='k', s=30, label=f'{long1}°')
    #ax.scatter(x_2, y_2, c='b', s=30, label=f'{long2}°')
    #r_ring = 2 #kpc
    #theta_ring = np.linspace(0, 2*np.pi, 100)
    #x_ring = r_ring*np.cos(theta_ring)
    #y_ring = r_ring*np.sin(theta_ring) + const.r_s
    #ax.plot(x_ring, y_ring, c='g', label='2 kpc ring')
    #add_sag_to_ax(ax, x_sa, y_sa)
    #plt.legend()
    #plt.xlabel('kpc')
    #plt.ylabel('kpc')
    #plt.show()
    #plt.close()
    rho_min = np.min([rho_closest1, rho_closest2])
    rho_max = np.max([rho_closest1, rho_closest2])
    return rho_min, rho_max


if __name__ == '__main__':
    rho_min, rho_max = find_rho_min_max(30, 55)
    print(rho_min, rho_max)