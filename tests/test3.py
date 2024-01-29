import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate coordinates for a sphere
def generate_sphere(radius, num_points):
    theta = np.linspace(0, np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    radians = np.linspace(0, radius, num_points)
    radians, theta, phi = np.meshgrid(radians, theta, phi)
    
    x = radians * np.sin(theta) * np.cos(phi)
    y = radians * np.sin(theta) * np.sin(phi)
    z = radians * np.cos(theta)
    
    return x.ravel(), y.ravel(), z.ravel()

# Generate sphere coordinates
radius = 3.0
num_points = 10
x, y, z = generate_sphere(radius, num_points)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o', label='Sphere')

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Set plot title
plt.title('3D Scatter Plot of Sphere')

# Show the plot
plt.legend()
plt.show()
