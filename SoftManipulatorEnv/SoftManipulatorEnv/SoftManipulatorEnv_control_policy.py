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
        
        self.soft_robot = SoftRobotBasicEnvironment(number_of_segment=1,gui=gui)
        self._base_link_id = None
        
        p0,o0 = ((0, 0, 0.1), (0, 0, 0, 1))
        
        p0,o0 = self.soft_robot.bullet.multiplyTransforms(p0, o0, [0.0, 0.0,0.00], [0,0,0,1])
        angle = -np.pi
        rotation_quaternion = self.soft_robot.bullet.getQuaternionFromEuler([0, 0, angle/2])
        
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
        
        
        self.soft_robot.move_robot_ori(action=np.array([0., 0.0, 0.0]),
                                base_pos = self.new_pos, base_orin = self.base_orin,camera_marker=False)
        
        self.soft_robot.bullet.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=35, cameraPitch=-30, cameraTargetPosition=[0,0,0])

        self.reset()
            
        ### IK
        self.action_space = spaces.Box(low=np.array([-0.015,-0.015]),
                                       high=np.array([0.015,0.015]), dtype="float32")
        
        observation_bound = np.array([1, 1, 1]) # target, pos, ori  
         
        self.observation_space = spaces.Box(low = -observation_bound, high = observation_bound, dtype="float32")
        
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def observe(self):        
        ob = self.desired_pos

        
        return ob

    def step(self, action):

        self._shape, self._ode_sol = self.soft_robot.move_robot_ori(action=np.array([0.0, action[0], action[1]]),
                                                                        base_pos = self.new_pos, base_orin = self.base_orin,camera_marker=False)
     
        
        self.pos = self._shape[-1][:3]
        self.distance = np.linalg.norm(self.desired_pos - self.pos)
        
        reward = (math.exp(-500*(self.distance**2))) 
        observation = self.observe()
        done = True
        abs_err = np.abs(self.desired_pos - self.pos)
        if self._gui:
            # print (f"rew:{reward:0.4f}")
            print (f"{abs_err[0]:0.4f},{abs_err[1]:0.4f},{abs_err[2]:0.4f},{self.distance:0.4f}")
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
        self._command_x  = 0*np.random.uniform(low=-0.015, high=0.015)
        self._command_y  = 0*np.random.uniform(low=-0.015, high=0.015)
        
        self._shape, self._ode_sol = self.soft_robot.move_robot_ori(action=np.array([0.0, self._command_x, self._command_y]),
                                base_pos = self.new_pos, base_orin = self.base_orin,camera_marker=False)
       
        self.desired_pos = self._shape[-1][:3] 
        
        if (self._gui): #Test env
            # self.soft_robot._set_marker(self.desired_pos)        
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



def get_ref(gt,traj_name='Circle'):
    
        if traj_name == 'Rose':
            k = 4
            T  = 200
            w  = 2*np.pi/T
            a = 0.025
            r  = a * np.cos(k*w*gt)
            xd = (x0 + np.array((r*np.cos(w*gt),r*np.sin(w*gt),0.00*gt)))
            xd_dot = np.array((-r*w*np.sin(w*gt),r*w*np.cos(w*gt),0.00*gt))
        elif traj_name == 'Limacon':
            T  = 100
            w  = 2*np.pi/T
            radius = 0.02
            radius2 = 0.03
            shift = -0.02
            xd = (x0 + np.array(((shift+(radius+radius2*np.cos(w*gt))*np.cos(w*gt)),(radius+radius2*np.cos(w*gt))*np.sin(w*gt),0.00*gt)))
            xd_dot = np.array((radius*(-w*np.sin(w*(gt)-0.5*w*np.sin(w/2*(gt)))),radius*(w*np.cos(w*(gt)-0.5*radius2*np.cos(w/2*gt))),0.00))                            
        elif traj_name=='Circle':
            T  = 50
            w  = 2*np.pi/T
            radius = 0.02
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.00*gt)))
            xd_dot = np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.00))
        elif traj_name=='Helix':
            T  = 50
            w  = 2*np.pi/T
            radius = 0.04
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.0001*gt)))
            xd_dot = ( np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.0001)))
        elif traj_name=='Eight_Figure':
            T  = 25*2
            A  = 0.02
            w  = 2*np.pi/T
            xd = np.array((A*np.sin(w*gt) , A*np.sin((w/2)*gt),0.1))
            xd_dot = np.array((A*w*np.cos(w*gt),A*w/2*np.cos(w/2*gt),0.00))
        elif traj_name=='Moving_Eight_Figure':
            T  = 25*2
            A  = 0.03
            w  = 2*np.pi/T
            xd = np.array(x0+(A*np.sin(w*gt) , A*np.sin((w/2)*gt),0.0002*gt))
            xd_dot = np.array((A*w*np.cos(w*gt),A*w/2*np.cos(w/2*gt),0.0002))
        elif traj_name=='Square':        
            T  = 12.5*2
            tt = gt % (4*T)
            scale = 2

            if (tt<T):
                xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,0.0)))
                xd_dot = scale*np.array(((0.02/T),0,0))
            elif (tt<2*T):
                xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.0)))
                xd_dot = scale*np.array((0,-(0.02/T),0))
            elif (tt<3*T):
                xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
                xd_dot = scale*np.array((-(0.02/T),0,0))
            elif (tt<4*T):
                xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.0)))
                xd_dot = scale*np.array((0,+(0.02/T),0))
            else:
                # t0 = time.time()+5
                gt = 0
        elif traj_name=='Moveing_Square':        
            T  = 10.0
            tt = gt % (4*T)
            if (tt<T):
                xd = (x0 + 2*np.array((-0.01+(0.02/T)*tt,0.01,-0.02+0.0005*gt)))
                xd_dot = 2*np.array(((0.02/T),0,0.0005))
            elif (tt<2*T):
                xd = (x0 + 2*np.array((0.01,0.01-((0.02/T)*(tt-T)),-0.02+0.0005*gt)))
                xd_dot = 2*np.array((0,-(0.02/T),0.0005))
            elif (tt<3*T):
                xd = (x0 + 2*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,-0.02+0.0005*gt)))
                xd_dot = 2*np.array((-(0.02/T),0,0.0005))
            elif (tt<4*T):
                xd = (x0 + 2*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),-0.02+0.0005*gt)))
                xd_dot = 2*np.array((0,+(0.02/T),0.0005))
              
        elif traj_name=='Triangle':        
            T  = 12.5 *2
            tt = gt % (4*T)
            scale = 2
            if (tt<T):
                xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,-0.01+(0.02/T)*tt,0.0)))
                xd_dot = scale*np.array(((0.02/T),(0.02/T),0))
            elif (tt<2*T):
                xd = (x0 + scale*np.array((0.01+(0.02/T)*(tt-(T)),0.01-((0.02/T)*(tt-(T))),0.0)))
                xd_dot = scale*np.array(((0.02/T),-(0.02/T),0))
            elif (tt<4*T):
                xd = (x0 + scale*np.array((0.03-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
                xd_dot = scale*np.array((-(0.02/T),0,0))
            else:
                # t0 = time.time()+5
                gt = 0
        else: # circle
            T  = 50*2
            w  = 2*np.pi/T
            radius = 0.02
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.00*gt)))
            xd_dot = np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.00))
            
        return xd,xd_dot




if __name__ =="__main__":
    
    Train = False
    num_cpu_core = 50 if Train else 1
    max_epc = 2000000


    if (num_cpu_core == 1):
        sf_env = SoftManipulatorEnv()
    else:
        sf_env = SubprocVecEnv([make_env(i, i) for i in range(1, num_cpu_core)]) # Create the vectorized environment
    
    
    if Train:
        timestr   = time.strftime("%Y%m%d-%H%M%S")
        logdir    = "logs/learnedPolicies/log_"  + timestr

        # model = SAC("MlpPolicy", sf_env, verbose=1, tensorboard_log=logdir)
        model = SAC.load("logs/learnedPolicies/model_20240611-102855", env = sf_env) # good model for 5 seg IK

        model.learn(total_timesteps=max_epc,log_interval=10)
        timestr   = time.strftime("%Y%m%d-%H%M%S")
        modelName = "logs/learnedPolicies/model_"+ timestr
        model.save(modelName)
        sf_env.close()
        print(f"finished. The model saved at {modelName}")
        
    else:
            
        model = SAC.load("logs/learnedPolicies/model_20240611-113029", env = sf_env) # good model for 5 seg IK
        obs = sf_env.reset()
        tf = 50
        ts = 0.01
        x0 = np.array([0.1,0,0.1])
        traj_name = 'Triangle'
        gt = 0
        # plot refrence trajectory 
        prevPose = x0
        for i in range(int(tf/(ts*10))):
            gt += (ts*10)
            xd, xd_dot = get_ref(gt,traj_name)
            
            # xd_1 = np.array([xd[1],xd[2],xd[0]])
            xd_1 = np.array([xd[2],xd[1],xd[0]])
            sf_env.soft_robot.bullet.addUserDebugLine(prevPose, xd_1, [0, 0, 0.3], 5, 0) 
            prevPose = xd_1
        
        prevPose = x0
        gt = 0
        
        for i in range(int(tf/ts)):
            xd, xd_dot = get_ref(gt,traj_name)
            sf_env.desired_pos = np.array([xd[2],xd[1],xd[0]])

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = sf_env.step(action)
          
            sf_env.soft_robot._dummy_sim_step(1)
            
            xcc = np.array([xd[2],xd[1],xd[0]])
            sf_env.soft_robot.bullet.addUserDebugLine(prevPose,  sf_env.pos, [1, 0, 0.3], 5, 0) 
            prevPose = sf_env.pos

                # sf_env._env._dummy_sim_step(1)
            
            # if done:
            #     time.sleep(1)            
            #     obs = sf_env.reset()
            #     time.sleep(0.1)
                # sf_env._env._dummy_sim_step(1)
                
            # if int((i*ts)*100)%10 == 0:
            #     xcc = np.array([xc[0],xc[2],xc[1]+0.1])
            #     env._pybullet.addUserDebugLine(prevPose, xcc, [1, 0, 0.3], 5, 0) 
            #     prevPose = xcc
            
            gt += ts
       
