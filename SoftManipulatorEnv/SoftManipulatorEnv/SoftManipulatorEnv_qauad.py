import numpy as np
import gym
from   gym import spaces
from   numpy.core.function_base import linspace

from   stable_baselines3.common.env_util import make_vec_env
from   stable_baselines3 import PPO, SAC
from   stable_baselines3.common.utils import set_random_seed
from   stable_baselines3.sac.policies import MlpPolicy
from   stable_baselines3.common.vec_env import SubprocVecEnv
from   stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from   stable_baselines3.common.callbacks import CheckpointCallback

import math
from random import random
import time

from scripts.CPG import CPG
from scripts.mini_spot_test import A1Env

from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment
import numpy as np


class SoftManipulatorEnv(gym.Env):
    def __init__(self,gui=True) -> None:
        super(SoftManipulatorEnv, self).__init__()

        self.simTime = 0
        self._gui  = gui 
        
        self._env = A1Env(gui=gui)
        self.soft_robot = SoftRobotBasicEnvironment(bullet = self._env.bullet,number_of_segment=3)
        # self.soft_robot = SoftRobotBasicEnvironment(number_of_segment=3,gui=gui)
        self._base_link_id = None
        
        # p0,o0 = self._env.get_ee_state()
        p0,o0 = ((0.017996429412476762, 0.0001648717198692931, 0.26283791331434025), (1.583553307424927e-05, -0.000778847170805682, 0.0007656098485339273, 0.9999994034937623))
        
        p0,o0 = self.soft_robot.bullet.multiplyTransforms(p0, o0, [0.23, 0.0,0.09], [0,0,0,1])
        angle = np.pi
        rotation_quaternion = self.soft_robot.bullet.getQuaternionFromEuler([angle, 0, angle/2])
        
        self.new_pos, self.new_ori = self.soft_robot.bullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
        self.base_orin = self.soft_robot.bullet.getEulerFromQuaternion(self.new_ori)
        if self._base_link_id is None:
            base_link_shape = self.soft_robot.bullet.createVisualShape(self.soft_robot.bullet.GEOM_BOX, halfExtents=[0.025, 0.025, 0.03], rgbaColor=[0.6, 0.6, 0.6, 1])
            base_link_pos, base_link_ori = self.soft_robot.bullet.multiplyTransforms(self.new_pos, self.new_ori, [0,-0.02,0], [0,0,0,1])
            self._base_link_id    = self.soft_robot.bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_link_shape,
                                                        baseVisualShapeIndex=base_link_shape,
                                                        basePosition= base_link_pos , baseOrientation=base_link_ori)
        else:
            base_link_pos, base_link_ori = self.soft_robot.bullet.multiplyTransforms(self.new_pos, self.new_ori, [0,-0.02,0.0], [0,0,0,1])
            self.soft_robot.bullet.resetBasePositionAndOrientation(self._base_link_id, base_link_pos , base_link_ori)
        
        self._initial_pos = [0.42,0.0,0.041]
        self.obj_id = self.soft_robot.add_a_cube(self._initial_pos,[0.1,0.1,0.1],mass=0.025,color=[1,0,1,1])

        self.soft_robot.move_robot_ori(action=np.array([0.08 ,0.0, 0.01, 
                                                      0., 0.0, 0.0,
                                                     0., 0.0, 0.0]),
                                base_pos = self.new_pos, base_orin = self.base_orin,camera_marker=False)
        
        
        
        # self._env.add_a_cube([0.02,0.0,0.2],[0.05,0.05,0.05],mass=0.01,color=[1,0,1,1])

        self.soft_robot.bullet.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=35, cameraPitch=-30, cameraTargetPosition=[0,0,0])

        self.reset()
            
        ### IK
        self.action_space = spaces.Box(low=np.array([-0.015,-np.pi*2,
                                                     -0.005,-np.pi*2,
                                                     -0.01,-np.pi*2,
                                                     -0.01,-np.pi*2,
                                                     -0.01,-np.pi*2,
                                                     -0.01,-np.pi*2]),
                                       high=np.array([0.015,np.pi*2,
                                                      0.005,np.pi*2,
                                                      0.01,np.pi*2,
                                                      0.005,np.pi*2,
                                                      0.01, np.pi*2,
                                                      0.005,np.pi*2]), dtype="float32")
        observation_bound = np.array([1, 1, 1]) # target, pos, ori  
         
        self.observation_space = spaces.Box(low = -observation_bound, high = observation_bound, dtype="float32")
        
        ### FK
        # self.action_space = spaces.Box(low=np.array([-0.02,-0.02,0.0]), high=np.array([0.2,0.2,0.2]), dtype="float32")
        # observation_bound = np.array([np.inf, np.inf, np.inf]) # l uy ux 

        # self.observation_space = spaces.Box(low = -observation_bound, high = observation_bound, dtype="float32")
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def observe(self):        
        # ob      = ([self.ol ,self.ouy, self.oux])
        # ob = np.concatenate ((self.desired_pos, np.array(self.obj_pose[0]),np.array(self.obj_pose[1])))
        ob = self.desired_pos
        
        return ob

    def step(self, action):

        
        touch = 0
        for i in range(200):
            t = i*0.01    
            cable_1   = action[0]*np.sin(action[1]*t)
            cable_2   = 0.013 + action[2]*np.sin(action[3]*t)
            
            cable_3   = action[4]*np.sin(action[5]*t)
            cable_4   = action[6]*np.sin(action[7]*t)            
            
            cable_5   = action[8]*np.sin(action[9]*t)
            cable_6   = action[10]*np.sin(action[11]*t)

                    
            last_tip_pos, _ = self.soft_robot.bullet.getBasePositionAndOrientation(self.soft_robot._robot_bodies[-3])

            self._shape, self._ode_sol = self.soft_robot.move_robot_ori(action=np.array([0.08, cable_1, cable_2, 
                                                                                         0., cable_3, cable_4,
                                                                                         0., cable_5, cable_6]),
                                                                        base_pos = self.new_pos, base_orin = self.base_orin,camera_marker=False)
            

            
            
            if self.soft_robot.is_tip_in_contact(self.obj_id):
                touch +=1
            
                pos1, _ = self.soft_robot.bullet.getBasePositionAndOrientation(self.soft_robot._robot_bodies[-3])
                vel1    = np.array(pos1) - np.array(last_tip_pos) / 0.01
                
                pos2, _ = self.soft_robot.bullet.getBasePositionAndOrientation(self.obj_id)
                direction = [pos2[i] - pos1[i] for i in range(3)]
                
                # Normalize the direction vector
                norm = sum(x**2 for x in direction) ** 0.5
                direction = [x / norm for x in direction]

                # Calculate the velocity magnitude
                velocity_magnitude = sum(v**2 for v in vel1) ** 0.5

                # Determine the force magnitude based on the velocity magnitude
                force_magnitude = 0.035 * (1 + velocity_magnitude)  # Example scaling function
                # Compute the force vector
                force = [force_magnitude * x for x in direction]

                self.soft_robot.bullet.applyExternalForce(self.obj_id, -1, force, [0, 0, 0],  self.soft_robot.bullet.WORLD_FRAME)
                self.soft_robot.wait(0.2)
                
                # print (f"force: {force}")
                    

                    
        self.obj_pos = np.array(self.soft_robot.bullet.getBasePositionAndOrientation(self.obj_id)[0])
        # self.obj_pos[2] = 0

        self.pos = self._shape[-1][:3]
        self.distance_obj = np.linalg.norm(self.desired_pos-self.obj_pos)
        # self.distance_obj_tip = np.linalg.norm(self.pos-self.obj_pos)

        reward = (math.exp(-300*(self.distance_obj**2))) + (0.5 if touch >0 else 0)
        observation = self.observe()
        done = True

        if self._gui:
            # print (f"rew:{reward:0.4f}")
            abs_err = np.abs(self.desired_pos - self.obj_pos)

            print (f"{abs_err[0]:0.4f},{abs_err[1]:0.4f},{self.distance_obj:0.4f}")
            self.soft_robot._dummy_sim_step(1)
        

        info = {}
        if done:
            info = {
                'episode': {
                    'r': reward,
                    'l': self.current_step
                }
            }
        return observation, reward, done, info


    def reset(self):
        
        self.current_step = 1

        self.soft_robot.move_robot_ori(action=np.array([0.08, 0, 0.013,  
                                                      0., 0.0, -0.00,
                                                     0., 0.0, 0.0]),
                                base_pos = self.new_pos, base_orin = self.base_orin,camera_marker=False)
        
        des_x  = np.random.uniform(low=0.55, high=0.7, size=(1,))
        des_y  = np.random.uniform(low=-0.1, high=0.1, size=(1,))
        des_z  = 0*np.random.uniform(low= 0.0, high=0.1, size=(1,))
        self.desired_pos = np.squeeze(np.array((des_x,des_y,des_z)))
        
        
        self.soft_robot.bullet.resetBasePositionAndOrientation(self.obj_id, self._initial_pos, [0,0,0,1])
        
        for i in range(10):
            self.soft_robot.bullet.stepSimulation()

        
        
        if (self._gui): #Test env
            self.soft_robot._set_marker(self.desired_pos)
        
            # print ("reset Env 0")
            
            self.soft_robot.wait(0.5)

        observation = self.observe()
        
        return observation  # reward, done, info can't be included

    def close (self):
        print ("Environment is closing....")



    
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = SoftManipulatorEnv(gui=False)
        #DummyVecEnv([lambda: CustomEnv()]) #gym.make(env_id)
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    set_random_seed(seed)
    return _init

if __name__ =="__main__":
    
    Train = False
    num_cpu_core = 60 if Train else 1
    max_epc = 1000000

    if (num_cpu_core == 1):
        sf_env = SoftManipulatorEnv()
    else:
        sf_env = SubprocVecEnv([make_env(i, i) for i in range(1, num_cpu_core)]) # Create the vectorized environment
        
    if Train:
        timestr   = time.strftime("%Y%m%d-%H%M%S")
        logdir    = "logs/learnedPolicies/log_"  + timestr

        model = SAC.load("logs/learnedPolicies/model_20240611-022827", env = sf_env) 
        model.learn(total_timesteps=max_epc,log_interval=10)
        timestr   = time.strftime("%Y%m%d-%H%M%S")
        modelName = "logs/learnedPolicies/model_"+ timestr
        model.save(modelName)
        sf_env.close()
        print(f"finished. The model saved at {modelName}")
        
    else:
            
        model = SAC.load("learnedPolicies/model_20240611-225749_best_quad", env = sf_env)         
        obs = sf_env.reset()
        timesteps = 50
        for i in range(timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sf_env.step(action)
            if done:
                time.sleep(1)            
                obs = sf_env.reset()
                time.sleep(0.1)
                # sf_env._env._dummy_sim_step(1)
