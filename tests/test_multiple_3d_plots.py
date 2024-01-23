import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.lines import Line2D


# Create a figure
fig = plt.figure(figsize=(12, 6))

# First subplot - 3D Surface Plot
ax1 = fig.add_subplot(121, projection='3d')
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
surface = ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('3D Surface Plot')

# Second subplot - 3D Scatter Plot
ax2 = fig.add_subplot(122, projection='3d')
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)
scatter = ax2.scatter(x, y, z, label='Scatter Plot')
ax2.set_title('3D Scatter Plot')
#ax2.legend()

# Show the plot
legend_exploded = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=4, label='Exploded')
legend_asd = Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=4, label='asd')
plt.legend(handles = [legend_exploded, legend_asd])
plt.suptitle('3D Scatter Plot')
plt.title("tagsydhuj")
plt.show()
