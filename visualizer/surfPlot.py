from cProfile import label
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

labelFontSize = 15
xytickSize = 13
legendFontSize = 13
DataPath1 = "logData/data_corl22_20220607-094247.dat"
DataPath2 = "logData/data_20220526-151634" # full dataset
data1 = np.loadtxt(DataPath1,dtype=np.float32,delimiter=',',comments='#')
data2 = np.loadtxt(DataPath2,dtype=np.float32,delimiter=',',comments='#')

X = data1[:,3]
Y = data1[:,4]
Z = data1[:,5] #.reshape(X.shape)
s = np.random.choice(np.arange(X.shape[0], dtype=np.int64), 1000, replace=False)



xs = X.reshape(60,60) 
ys = Y.reshape(60,60) 
zs = Z.reshape(60,60) 

# Plot the surface.

surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# fake2Dline = ax.plot([0],[0],[0.1], linestyle="none", c='r', marker = 'o',label="Training space")
# ax.legend([fake2Dline], ['Lyapunov function on XY plane'], numpoints = 1)
# x = X.reshape


# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/2:10j]
x = 0.05*(np.cos(u)*np.sin(v))
y = 0.05*(np.sin(u)*np.sin(v))
z = 0.05+0.1*np.cos(v)
ax.plot_wireframe(x, y, z, color=[0,0,0.8,0.5],label='Feasible Kinematics space')

ax.set_xlabel("X [m]",fontsize = labelFontSize)
ax.set_ylabel("Y [m]",fontsize = labelFontSize)
ax.set_zlabel("Z [m]",fontsize = labelFontSize)
ax.grid()
# ax.legend()
ax.legend(loc='upper center',ncol=1, bbox_to_anchor=(0.8, 0.8),fontsize = legendFontSize,)

plt.tick_params(which='minor', length=4, color='r')
plt.xticks(fontsize=xytickSize)
plt.yticks(fontsize=xytickSize)



# ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)





# ax.scatter(X[s],Y[s],Z[s])


# X = data1[:,3]
# Y = data1[:,4]
# Z = data1[:,5] #.reshape(X.shape)
# s = np.random.choice(np.arange(X.shape[0], dtype=np.int64), 80, replace=False)
# ax.scatter(X[s],Y[s],Z[s])

# # Customize the z axis.
# ax.set_zlim(.0, .15)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# # fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()