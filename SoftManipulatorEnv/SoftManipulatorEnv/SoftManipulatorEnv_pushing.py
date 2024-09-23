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



from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment
import numpy as np


class SoftManipulatorEnv(gym.Env):
    def __init__(self,gui=True) -> None:
        super(SoftManipulatorEnv, self).__init__()

        self.simTime = 0
        self._gui  = gui
        
        
        self._env = SoftRobotBasicEnvironment(body_sphere_radius=0.02,number_of_segment=3,gui=self._gui)
        base_link_shape = self._env.bullet.createVisualShape(self._env.bullet.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[0.6, 0.6, 0.6, 1])
        base_link_pos, base_link_ori = self._env.bullet.multiplyTransforms([0,0,0.51], [0,0,0,1], [0,-0.0,0], [0,0,0,1])
        base_link_id    = self._env.bullet.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=base_link_shape,
                                                            baseVisualShapeIndex=base_link_shape,
                                                            basePosition= base_link_pos , baseOrientation=base_link_ori)
        self._base_pos = np.array([0,0,0.5])
        self._base_ori = np.array([-np.pi/2,0,0])
        shape, ode_sol = self._env.move_robot_ori(action=np.array([0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0]),
                                            base_pos=self._base_pos, base_orin = self._base_ori, camera_marker=False)
        
        # self._env.add_a_cube([0.,0.0,0.01],[0.2,0.2,0.1],mass=1,color=[0.2,0.2,0.2,1])
        self._initial_pos = [0.1,0.0,0.041]
        self.obj_id = self._env.add_a_cube(self._initial_pos,[0.08,0.08,0.08],mass=0.05,color=[1,0,1,1])

        self._env.bullet.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=35, cameraPitch=-30, cameraTargetPosition=[0,0,0])

        self.reset()
            
        ### IK
        self.action_space = spaces.Box(low=np.array([-0.02,-5,
                                                     -0.02,-5,
                                                     -0.02,-5,
                                                     -0.02,-5]),
                                       high=np.array([0.02,5,
                                                      0.02,5,
                                                      0.02,5,
                                                      0.02,5]), dtype="float32")
        
        observation_bound = np.array([1, 1, 1]) # pos 
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
        ob = self.desired_pos
        return ob

    def step(self, action):

        touch  = 0
        for i in range(50):
            t = i*0.005    
            seg1_cable_1   = action[0]*np.sin(action[1]*np.pi*t)
            seg1_cable_2   = action[2]*np.sin(action[3]*np.pi*t)
            
            seg2_cable_1   = action[4]*np.sin(action[5]*np.pi*t)
            seg2_cable_2   = action[6]*np.sin(action[7]*np.pi*t)
            
            self._shape, self._ode_sol = self._env.move_robot_ori(action=np.array([0.0, seg1_cable_1, seg1_cable_2,
                                                                                0.005, seg2_cable_1, seg2_cable_2]),
                                                base_pos=self._base_pos, base_orin = self._base_ori, camera_marker=False)
            
            
            if self._env.is_tip_in_contact(self.obj_id):
                # Calculate the direction from object1 to object2
                touch += 1
                self._env.apply_force(force_magnitude = 1, obj_id = self.obj_id)
                # pos1, _ = self._env.bullet.getBasePositionAndOrientation(self._env._robot_bodies[-3])
                # pos2, _ = self._env.bullet.getBasePositionAndOrientation(self.obj_id)
                # direction = [pos2[i] - pos1[i] for i in range(3)]
                
                # # Normalize the direction vector
                # norm = sum(x**2 for x in direction) ** 0.5
                # direction = [x / norm for x in direction]
                
                # force_magnitude = 1  # Adjust this value as needed
                # force = [force_magnitude * x for x in direction]
                # self._env.bullet.applyExternalForce(self.obj_id, -1, force, [0, 0, 0],  self._env.bullet.WORLD_FRAME)
                # self._env.bullet.stepSimulation()
                    
        self.obj_pos = np.array(self._env.bullet.getBasePositionAndOrientation(self.obj_id)[0])

        self.pos = self._shape[-1][:3]
        self.distance_obj = np.linalg.norm(self.desired_pos-self.obj_pos)
        # self.distance_obj_tip = np.linalg.norm(self.pos-self.obj_pos)

        reward = (math.exp(-50*(self.distance_obj**2))) + (0.5 if touch > 0 else 0)
        
        observation = self.observe()
        done = True
        if self._gui:
            print (f"rew:{reward:0.4f}")
            self._env._dummy_sim_step(1)
        

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

        self._shape, self._ode_sol = self._env.move_robot_ori(action=np.array([0.0, 0.0, 0.0,
                                                                               0.0, 0.0, 0.0,
                                                                               0.0, 0.0, 0.0,
                                                                               0.0, 0.0, 0.0,
                                                                               0.0, 0.0, 0.0]),
                                                base_pos=self._base_pos, base_orin = self._base_ori, camera_marker=False)
            
        des_x  = np.random.uniform(low=0.1, high=0.2, size=(1,))
        des_y  = np.random.uniform(low=-0.3, high=0.3, size=(1,))
        des_z  = 0*np.random.uniform(low= 0.0, high=0.1, size=(1,))
        self.desired_pos = np.squeeze(np.array((des_x,des_y,des_z)))
        self._env.bullet.resetBasePositionAndOrientation(self.obj_id, self._initial_pos, [0,0,0,1])
        
        for i in range(10):
            self._env.bullet.stepSimulation()

        

        # self.ol    = np.random.uniform(low= -0.03, high=0.04, size=(1,))[0]
        # self.ouy   = np.random.uniform(low=-0.015, high=0.015, size=(1,))[0]
        # self.oux   = np.random.uniform(low=-0.015, high=0.015, size=(1,))[0]
        
        # self.ol    = np.random.uniform(low= -0.01, high=0.01, size=(1,))[0]
        # self.ouy   = np.random.uniform(low=-0.005, high=0.005, size=(1,))[0]
        # self.oux   = np.random.uniform(low=-0.005, high=0.005, size=(1,))[0]
        
        # self.ol    = np.random.uniform(low= -0.005, high=0.005, size=(1,))[0]
        # self.ouy   = np.random.uniform(low=-0.003, high=0.003, size=(1,))[0]
        # self.oux   = np.random.uniform(low=-0.003, high=0.003, size=(1,))[0]
        
        
        
        if (self._gui): #Test env
            self._env._set_marker(self.desired_pos)
        
            print ("reset Env 0")
    
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
    
    num_cpu_core = 1
    max_epc = 500000
    
    # from gym.envs.registration import register
    # register(
    #     id='SoftManipulatorEnv-v0',
    #     entry_point='custom_env:SoftManipulatorEnv',
    # )

    if (num_cpu_core == 1):
        sf_env = SoftManipulatorEnv()
    else:
        sf_env = SubprocVecEnv([make_env(i, i) for i in range(1, num_cpu_core)]) # Create the vectorized environment
    
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    logdir    = "logs/learnedPolicies/log_"  + timestr

    model = SAC("MlpPolicy", sf_env, verbose=1, tensorboard_log=logdir)

    # model.load("logs/learnedPolicies/model_20240603-012421.zip")
    
    
    model.learn(total_timesteps=max_epc,log_interval=10)
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    modelName = "logs/learnedPolicies/model_"+ timestr
    model.save(modelName)
    sf_env.close()
    print(f"finished. The model saved at {modelName}")
    
    
        
    # # model = SAC.load("logs/learnedPolicies/model_20240603-085205.zip", env = sf_env)
    # model = SAC.load("logs/learnedPolicies/model_20240605-070914", env = sf_env)

    
    # obs = sf_env.reset()
    # timesteps = 5000
    # for i in range(timesteps):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = sf_env.step(action)
    #     #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-99.0, verbose=1)
    #     #eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
    #     if done:
    #         time.sleep(1)
            
    #         obs = sf_env.reset()
    #         time.sleep(0.1)
    #         sf_env._env._dummy_sim_step(1)

    # obs = sf_env.reset()
    # timesteps = 5000000
    # for i in range(timesteps):
    #     t = i*0.005
    #     sf1_seg1_cable_1   = .005*np.sin(0.05*np.pi*t)
    #     obs, reward, done, info = sf_env.step(np.array([sf1_seg1_cable_1,0.0,0.01,0.0,0.002,0.0,0.0,0.0,0.0,0.0,0.0]))
        
    #     #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-99.0, verbose=1)
    #     #eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
    #     # if done:
    #     #     time.sleep(1)
            
    #         # obs = sf_env.reset()
    #         # time.sleep(0.1)
    #         # sf_env._env._dummy_sim_step(1)

