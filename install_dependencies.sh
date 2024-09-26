#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install numpy, scipy, matplotlib and collection
pip install numpy scipy matplotlib collection


# Install pybullet
pip install pybullet

# Install stable-baselines3
pip install stable-baselines3

# Install getkey
pip install getkey

# Install open cv2
pip install opencv-python


# Install pill
pip install pillow





echo "All dependencies installed successfully!"
