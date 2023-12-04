import numpy as np
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.6f} seconds to run.")
        return result
    return wrapper
def count_negative_values(arr):
    negative_values = arr[arr < 0]
    return len(negative_values), np.average(negative_values)

""" test = np.ones((4,4,4)) * np.array([1,2,3,4]).T
test[1]*=2
test[2]*=3
test[3]*=4

np.save('test.npy', test)



test2 = np.load('test.npy')
test2[2] = 0
np.save('test.npy', test2)

test3 = np.load('test.npy')
a = np.random.randint(19, size=(1,4,4))
test3 = np.concatenate((test3, a), axis=0)
print(test3)
np.save('test.npy', test3)
test4 = np.load('test.npy')
print(test4)
 """

@timing_decorator
def test_time_draw_postitions_from_entire_array():
    # test_time_draw_postitions_from_entire_array took 1083.423133 seconds to run.
    data = np.load('output\galaxy_data\interpolated_densities.npy')
    x = np.load('output/galaxy_data/x_grid.npy')
    data = data/np.sum(data)
    for _ in range(1000):
        x = np.random.choice(a=len(data), size=1, p=data)

@timing_decorator
def test_time_draw_positions_rad_long_lat():
    #test_time_draw_positions_rad_long_lat took 0.900275 seconds to run.
    densities_longitudinal = np.load('output\galaxy_data\densities_longitudinal.npy')
    densities_longitudinal = densities_longitudinal/np.sum(densities_longitudinal)
    densities_lat = np.load('output\galaxy_data\densities_lat.npy')
    densities_lat = densities_lat/np.sum(densities_lat, axis=1, keepdims=True)
    rad_densities = np.load('output\galaxy_data\densities_rad.npy')
    rad_densities = rad_densities/np.sum(rad_densities, axis=0, keepdims=True)
    print(rad_densities.shape)
    print(rad_densities[:,0,0].shape)
    print(densities_longitudinal.shape)
    print(densities_lat.shape)
    print(np.sum(densities_lat, axis=1, keepdims=True).shape)
    print(densities_lat[0].ravel())
    for _ in range(1000):
        l = np.random.choice(a=len(densities_longitudinal), size=1, p=densities_longitudinal )
        b = np.random.choice(a=len(densities_lat[0]), size=1, p=densities_lat[l].ravel() )
        r = np.random.choice(a=len(rad_densities[:,0,0]), size=1, p=rad_densities[:, l, b].ravel() )
        #y = np.random.choice(a=len(densities_lat[0]), size=1, p=densities_lat[x].ravel() )
        #z = np.random.choice(a=len(all_densities[0]), size=1, p=all_densities[x][y].ravel() )
    

test_time_draw_positions_rad_long_lat()