#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install numpy, scipy, matplotlib and collection
pip install numpy==1.26.3
pip install scipy==1.11.4
pip install matplotlib==3.8.2
pip install collection

# Install pybullet
pip install pybullet==3.2.6

# Install stable-baselines3
pip install stable-baselines3==2.3.2

# Install getkey
pip install getkey==0.6.5

# Install open cv2
pip install opencv-python==4.9.0.80


# Install pill
pip install pillow





echo "All dependencies installed successfully!"
