# SoftManiSim
![alt](readme_assets/img.gif)
SoftManiSim is an advanced, open-source simulation framework designed for multi-segment continuum manipulators, offering a significant leap forward in accuracy and flexibility. Unlike traditional simulators, which often rely on simplifying assumptions like constant curvature bending and omission of contact forces, SoftManiSim employs a robust and rapid mathematical model to achieve precise, real-time simulations. Its versatility allows seamless integration with various rigid-body robots, making it applicable across a broad range of robotic platforms. Additionally, SoftManiSim supports parallel operations, enabling the simultaneous simulation of multiple robots and generating essential synthetic data for training deep reinforcement learning models. 



# Installation and Setup

## Clone the Repository:

```
git clone git@github.com:MohammadKasaei/SoftManiSim.git
cd SoftManiSim
```
## Set Up a Virtual Environment (optional):

```
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
## Install Dependencies:
Before running the script, make sure you have execute permissions. Run the following command:
```
chmod +x install_dependencies.sh
```
To install all the dependencies, simply run:
```
./install_dependencies.sh
```
Wait for the script to complete. Once done, all the required dependencies should be installed in your environment.



