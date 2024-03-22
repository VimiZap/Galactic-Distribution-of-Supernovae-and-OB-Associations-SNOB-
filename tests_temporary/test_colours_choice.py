import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Example data
x = np.linspace(0, 10, 100)
y_components = [np.sin(x) * (i+1) for i in range(7)]

# Define a list of 7 distinct colors
#colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink']
#colors = ['red', 'green', 'blue', 'c', 'm', 'lime', 'darkorange']
colors = sns.color_palette('bright', 7)

# Create the plot and assign each component a different color
plt.figure(figsize=(10, 6))
for i, y in enumerate(y_components):
    plt.plot(x, y, color=colors[i], label=f'Component {i+1}')

# Add a legend to the plot
plt.legend()

# Display the plot
plt.show()
