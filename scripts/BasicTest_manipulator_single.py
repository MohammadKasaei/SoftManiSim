import numpy as np
import time

from environment.BasicEnvironment import BasicEnvironment
from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment



def Jac(f, q, dq=np.array((1e-4,1e-4,1e-4,1e-4,1e-4,1e-4))):
    
    fx0 = f(q)
    n   = len(q)
    m   = len(fx0)
    jac = np.zeros((n, m))
    for j in range(n):  # through rows 
        if (j==0):
            Dq = np.array((dq[0]/2.0,0,0,0,0,0))
        elif (j==1):
            Dq = np.array((0,dq[1]/2.0,0,0,0,0))
        elif (j==2):
            Dq = np.array((0,0,dq[2]/2.0,0,0,0))
        elif (j==3):
            Dq = np.array((0,0,0,dq[2]/2.0,0,0))
        elif (j==4):
            Dq = np.array((0,0,0,0,dq[2]/2.0,0))
        elif (j==5):
            Dq = np.array((0,0,0,0,0, dq[2]/2.0))
            
        jac [j,:] = (f(q+Dq) - f(q-Dq))/dq[j]
    return jac    



if __name__ == "__main__":
    
    env = BasicEnvironment()
    env.move_arm(target_pos= np.array([0.4,0.,0.35]), target_ori=[np.pi/2,np.pi/2,0],duration=0.01)
    env.wait(1)
    soft_robot_1 = SoftRobotBasicEnvironment(bullet= env._pybullet,number_of_segment=1)
    soft_robot_2 = SoftRobotBasicEnvironment(bullet= env._pybullet,head_color=[0,0.75,0,1])
     
    t = 0
    dt = 0.01
    while True:    
        t += dt
        # print (t)
        # if int(t*100) % 100 == 0:
        #     soft_env.in_hand_camera_capture_image()

        pos = np.array([
            0.3 + 0.05 * np.sin(0.1*np.pi*t),
            0.0 + 0.1 * np.sin(0.1*np.pi*t),
            0.5 + 0.03 * np.sin(0.1*np.pi*t)
        ])
        ori = 0*np.array([
            1.5 * np.sin(0.2*np.pi*t),
            np.pi/2 - 0.2 * np.sin(0.02*np.pi*t),
            0.0 * np.sin(0.02*np.pi*t),
        ])
    
        
        sf1_seg1_cable_1   = .005*np.sin(0.5*np.pi*t)
        sf1_seg1_cable_2   = .005*np.sin(0.5*np.pi*t)
        sf1_seg2_cable_1   = .005*np.sin(0.5*np.pi*t+1)
        sf1_seg2_cable_2   = .005*np.sin(0.5*np.pi*t+1)
        sf1_seg3_cable_0   = .00*np.sin(0.5*np.pi*t)
        sf1_seg3_cable_1   = .005*np.sin(0.5*np.pi*t+2)
        sf1_seg3_cable_2   = .005*np.sin(0.5*np.pi*t+2)
        sf1_gripper_pos    = np.abs(np.sin(np.pi*t))
        
       
        
        env.move_arm (target_pos= pos, target_ori=ori)
        
        p0,o0 = env.get_ee_state()
        p0,o0 = env._pybullet.multiplyTransforms(p0, o0, [0.025,-0.0,-0.0], [0,0,0,1])
        angle = -np.pi/2  # 90 degrees in radians
        rotation_quaternion = env._pybullet.getQuaternionFromEuler([0, 0, angle])
        new_pos, new_ori = env._pybullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
        base_orin = env._pybullet.getEulerFromQuaternion(new_ori)
        
        soft_robot_1.move_robot_ori(action=np.array([0.0, sf1_seg1_cable_1, sf1_seg1_cable_2, 
                                                 0.0, sf1_seg2_cable_1, sf1_seg2_cable_2,
                                                 sf1_seg3_cable_0, sf1_seg3_cable_1, sf1_seg3_cable_2]),
                                base_pos = new_pos, base_orin = base_orin)
        
        env.wait(0.2)
        
        
