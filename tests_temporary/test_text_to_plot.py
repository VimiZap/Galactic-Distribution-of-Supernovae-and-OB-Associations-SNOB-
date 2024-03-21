import numpy as np
import matplotlib.pyplot as plt

# Generate a sine wave
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)

# Place text along the curve
text_positions = np.linspace(0.5, 2 * np.pi - 0.5, 4)  # adjust points along the curve
for pos in text_positions:
    # Calculate the slope of the curve to approximate text rotation
    slope = np.cos(pos)  # derivative of sin is cos
    angle = np.degrees(np.arctan(slope))
    
    plt.text(pos, np.sin(pos), 'Spiral Arm', rotation=angle, rotation_mode='anchor', ha='center', va='center')

plt.show()
