# SoftManiSim
![alt](readme_assets/img.gif)

**SoftManiSim** is an advanced, open-source simulation framework designed for multi-segment continuum manipulators, offering a significant leap forward in accuracy and flexibility. Unlike traditional simulators, which often rely on simplifying assumptions like constant curvature bending and omission of contact forces, SoftManiSim employs a robust and rapid mathematical model to achieve precise, real-time simulations. Its versatility allows seamless integration with various rigid-body robots, making it applicable across a broad range of robotic platforms. Additionally, SoftManiSim supports parallel operations, enabling the simultaneous simulation of multiple robots and generating essential synthetic data for training deep reinforcement learning models. 



# Installation and Setup

## Clone the Repository:

```
git clone git@github.com:MohammadKasaei/SoftManiSim.git
cd SoftManiSim
```
## Set Up a Virtual Environment (optional):

```
python3 -m venv env
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


### Running Example Scripts

The `scripts` folder contains several example scripts to help you get started. You can run any of these scripts using the following command:

```bash
python3 -m scripts.SCRIPT_NAME
```
For example, to run the `BasicTest_manipulator_two_robot`, use:

```bash
python3 -m scripts.BasicTest_manipulator_two_robot script
```

Additionally, we have developed a set of Gym Environment, for more detail check the `SoftManipulatorEnv/SoftManipulatorEnv` folder.



### API Documentation

Below is the API documentation for the `SoftManiSim` class, detailing its essential methods, their arguments, and functionalities:

| **Method**                | **Argument**                           | **Description**                                                                                                                                 |
|---------------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `__init__`                | `bullet`                               | Optional physics engine instance, defaults to `None`, initializes PyBullet if not provided.                                                     |
|                           | `body_color`                           | RGBA color for the robot's body.                                                                                                                |
|                           | `head_color`                           | RGBA color for the robot's head.                                                                                                                |
|                           | `body_sphere_radius`                   | Radius of spheres used to build the robot’s body.                                                                                               |
|                           | `number_of_sphere`                     | Number of spheres constructing the robot’s body.                                                                                                |
|                           | `number_of_segment`                    | Number of segments in the robot's body.                                                                                                         |
|                           | `gui`                                  | Boolean to toggle the graphical interface, defaults to `True`.                                                                                   |
| `create_robot`            | -                                      | No arguments, sets up the robot's physical structure within the simulation. This function is invoked at the end of the constructor.              |
| `move_robot_ori`          | `action`                               | Array of actions defining movement commands for robot segments.                                                                                 |
|                           | `base_pos`                             | The base position of the robot in the simulation space.                                                                                          |
|                           | `base_orin`                            | The base orientation of the robot, specified as Euler angles.                                                                                    |
|                           | `camera_marker`                        | Boolean to display camera markers, defaults to `True`.                                                                                           |
| `calc_tip_pos`            | `action`                               | Array of actions affecting the tip’s position and orientation.                                                                                   |
|                           | `base_pos`                             | The base position from which the tip's calculations start.                                                                                       |
|                           | `base_orin`                            | Base orientation affecting the tip’s calculation.                                                                                                |
| `capture_image`           | `removeBackground`                     | Boolean to decide whether to remove the background from the image, defaults to `False`.                                                          |
| `in_hand_camera_capture_image` | -                              | No arguments, captures an image from the robot’s in-hand camera.                                                                                 |
| `is_robot_in_contact`     | `obj_id`                               | Object ID to check for contact with the robot.                                                                                                   |
| `is_gripper_in_contact`   | `obj_id`                               | Object ID to check for contact with the robot's gripper.                                                                                         |
| `suction_grasp`           | `enable`                               | Boolean to enable or disable the suction grasp mechanism.                                                                                        |
| `set_grasp_width`         | `grasp_width_percent`                  | Percentage of maximum grasp width to set for the gripper.                                                                                        |
| `add_a_cube`              | `pos`                                  | Position to place the cube in the simulation.                                                                                                    |
|                           | `ori`                                  | Orientation of the cube, given as a quaternion.                                                                                                  |
|                           | `size`                                 | Dimensions of the cube.                                                                                                                          |
|                           | `mass`                                 | Mass of the cube.                                                                                                                                |
|                           | `color`                                | RGBA color of the cube.                                                                                                                          |
|                           | `textureUniqueId`                      | Optional texture ID for the cube's surface.                                                                                                      |
| `wait`                    | `sec`                                  | Duration in seconds to delay the simulation.                                                                                                     |

This table summarizes the main methods and their functionalities for users who want to integrate or work with the `SoftManiSim` class in their projects.


# Citation
If you find our paper or this repository helpful, please cite our work:

```
@inproceedings{kasaeisoftmanisim,
  title={SoftManiSim: A Fast Simulation Framework for Multi-Segment Continuum Manipulators Tailored for Robot Learning},
  author={Kasaei, Mohammadreza and Kasaei, Hamidreza and Khadem, Mohsen},
  booktitle={8th Annual Conference on Robot Learning}
}
```

# License
This project is licensed under the MIT License.

# Acknowledgments
This work is supported by the Medical Research Council [MR/T023252/1].


