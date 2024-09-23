import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a new figure
fig = plt.figure()

# Set the 3D projection
ax = fig.add_subplot(111, projection='3d')

# Set the sphere's radius
r = 1

# Define the sphere's coordinates
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:10j]
x = r*np.cos(u)*np.sin(v)
y = r*np.sin(u)*np.sin(v)
z = r*np.cos(v)

# Plot the sphere
ax.plot_surface(x, y, z, color='b')

# Show the plot
plt.show()
