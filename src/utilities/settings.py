
# the interpolation is very memory intensive, so we may need to divide the grid into smaller parts to avoid memory errors
# note the larger num_grid_subdivisions is the less memory is needed per interpolation, but the runtime increases
# the scipy.interpolate.griddata function is used to interpolate the density distribution, and is called a total of 4 * num_grid_subdivisions times
# num_grid_subdivisions has to be minium 1. If a smaller value is used, ValueError is raised
num_grid_subdivisions = 4 
