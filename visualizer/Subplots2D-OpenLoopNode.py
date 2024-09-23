from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import math

from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm


figure, axis = plt.subplots(2, 4,frameon=False)
# add a subplot with no frame
labelFontSize = 15
legendFontSize = 12
bbox_to_anchor_pos = (0.5,1.11)

# ideal data
# DataPath1 = "neuralODE/logs/log_20220606-170216.dat"
# DataPath2 = "neuralODE/logs/log_20220606-170605.dat"
# DataPath3 = "neuralODE/logs/log_20220606-170419.dat"
# DataPath4 = "neuralODE/logs/log_20220606-165848.dat"

#  noisy with filtering data
DataPath1 = "neuralODE/logs/log_20220606-172953.dat"
DataPath2 = "neuralODE/logs/log_20220606-173131.dat"
DataPath3 = "neuralODE/logs/log_20220606-173312.dat"
DataPath4 = "neuralODE/logs/log_20220606-173443.dat"

# circle
data = np.loadtxt(DataPath1,dtype=np.float32,delimiter=' ',comments='#')
xc = data[:,2:5]
xd = data[:,5:8]


axis[0, 0].plot(xc[:,0], xc[:,1],'r',lw=2,label="Actual")
axis[0, 0].plot(xd[:,0], xd[:,1],'k--',lw=2,label="Refrence")
# axis[0, 0].set_title("Cosine Function")
axis[0, 0].set_xlabel("Position [m]",fontsize = labelFontSize)
axis[0, 0].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[0, 0].grid()
axis[0, 0].legend(loc='upper center',fontsize =legendFontSize, ncol=2, bbox_to_anchor=bbox_to_anchor_pos)

axis[1, 0].plot(data[:,0],xc[:,0],'r',lw=2,label="X")
axis[1, 0].plot(data[:,0],xd[:,0],'m--',lw=2,label="Xr")

axis[1, 0].plot(data[:,0],xc[:,1],'g',lw=2,label="Y")
axis[1, 0].plot(data[:,0],xd[:,1],'y--',lw=2,label="Yr")

axis[1, 0].plot(data[:,0],xc[:,2],'b',lw=2,label="Z")
axis[1, 0].plot(data[:,0],xd[:,2],'c--',lw=2,label="Zr")

# axis[1, 0].set_title("Cosine Function")
axis[1, 0].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[1, 0].set_xlabel("Time [s]",fontsize = labelFontSize)
axis[1, 0].grid()
axis[1, 0].legend(loc='upper center',ncol=3, bbox_to_anchor=(0.5, 0.8))

##############################################################################
data = np.loadtxt(DataPath2,dtype=np.float32,delimiter=' ',comments='#')
xc = data[:,2:5]
xd = data[:,5:8]

axis[0, 1].plot(xc[:,0], xc[:,1],'r',lw=2,label="Actual")
axis[0, 1].plot(xd[:,0], xd[:,1],'k--',lw=2,label="Refrence")
# axis[0, 0].set_title("Cosine Function")
axis[0, 1].set_xlabel("Position [m]",fontsize = labelFontSize)
# axis[0, 1].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[0, 1].grid()
# axis[0, 1].legend(loc='upper center',fontsize =legendFontSize, ncol=2, bbox_to_anchor=bbox_to_anchor_pos)

axis[1, 1].plot(data[:,0],xc[:,0],'r',lw=2,label="X")
axis[1, 1].plot(data[:,0],xd[:,0],'m--',lw=2,label="Xr")

axis[1, 1].plot(data[:,0],xc[:,1],'g',lw=2,label="Y")
axis[1, 1].plot(data[:,0],xd[:,1],'y--',lw=2,label="Yr")

axis[1, 1].plot(data[:,0],xc[:,2],'b',lw=2,label="Z")
axis[1, 1].plot(data[:,0],xd[:,2],'c--',lw=2,label="Zr")

# axis[1, 0].set_title("Cosine Function")
# axis[1, 1].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[1, 1].set_xlabel("Time [s]",fontsize = labelFontSize)
axis[1, 1].grid()
# axis[1, 1].legend(loc='upper center',ncol=3, bbox_to_anchor=(0.5, 0.8))
##############################################################################
data = np.loadtxt(DataPath3,dtype=np.float32,delimiter=' ',comments='#')
xc = data[:,2:5]
xd = data[:,5:8]

axis[0, 2].plot(xc[:,0], xc[:,1],'r',lw=2,label="Actual")
axis[0, 2].plot(xd[:,0], xd[:,1],'k--',lw=2,label="Refrence")
# axis[0, 0].set_title("Cosine Function")
axis[0, 2].set_xlabel("Position [m]",fontsize = labelFontSize)
# axis[0, 1].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[0, 2].grid()
# axis[0, 2].legend(loc='upper center',fontsize =legendFontSize, ncol=2, bbox_to_anchor=bbox_to_anchor_pos)

axis[1, 2].plot(data[:,0],xc[:,0],'r',lw=2,label="X")
axis[1, 2].plot(data[:,0],xd[:,0],'m--',lw=2,label="Xr")

axis[1, 2].plot(data[:,0],xc[:,1],'g',lw=2,label="Y")
axis[1, 2].plot(data[:,0],xd[:,1],'y--',lw=2,label="Yr")

axis[1, 2].plot(data[:,0],xc[:,2],'b',lw=2,label="Z")
axis[1, 2].plot(data[:,0],xd[:,2],'c--',lw=2,label="Zr")

# axis[1, 0].set_title("Cosine Function")
# axis[1, 1].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[1, 2].set_xlabel("Time [s]",fontsize = labelFontSize)
axis[1, 2].grid()
# axis[1, 2].legend(loc='upper center',ncol=3, bbox_to_anchor=(0.5, 0.8))



##############################################################################
data = np.loadtxt(DataPath4,dtype=np.float32,delimiter=' ',comments='#')
xc = data[:,2:5]
xd = data[:,5:8]

axis[0, 3].plot(xc[:,0], xc[:,1],'r',lw=2,label="Actual")
axis[0, 3].plot(xd[:,0], xd[:,1],'k--',lw=2,label="Refrence")
# axis[0, 0].set_title("Cosine Function")
axis[0, 3].set_xlabel("Position [m]",fontsize = labelFontSize)
# axis[0, 1].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[0, 3].grid()
# axis[0, 3].legend(loc='upper center',fontsize =legendFontSize, ncol=2, bbox_to_anchor=bbox_to_anchor_pos)

axis[1, 3].plot(data[:,0],xc[:,0],'r',lw=2,label="X")
axis[1, 3].plot(data[:,0],xd[:,0],'m--',lw=2,label="Xr")

axis[1, 3].plot(data[:,0],xc[:,1],'g',lw=2,label="Y")
axis[1, 3].plot(data[:,0],xd[:,1],'y--',lw=2,label="Yr")

axis[1, 3].plot(data[:,0],xc[:,2],'b',lw=2,label="Z")
axis[1, 3].plot(data[:,0],xd[:,2],'c--',lw=2,label="Zr")

# axis[1, 0].set_title("Cosine Function")
# axis[1, 1].set_ylabel("Position [m]",fontsize = labelFontSize)
axis[1, 3].set_xlabel("Time [s]",fontsize = labelFontSize)
axis[1, 3].grid()
# axis[1, 3].legend(loc='upper center',ncol=3, bbox_to_anchor=(0.5, 0.8))


plt.show()
# plt.savefig("traj.pdf", dpi=150)