import time
import numpy as np
import pybullet as p
import pybullet_data
import sys
import cv2
import random

from collections import namedtuple
from operator import methodcaller

from environment.camera.camera import Camera
from PIL import Image
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import CubicSpline

class BasicEnvironment():
    def __init__(self,urdf="ur5_suction") -> None:
        self._simulationStepTime = 0.01
        self.vis = True

        self.bullet = p
        p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self._simulationStepTime)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=-15, cameraPitch=-25, cameraTargetPosition=[0.2,0,0.3])
        self._pybullet = p

        # sim_arg = {'gui':'True'}
        self.plane_id = p.loadURDF('plane.urdf')
        
        # self.load_robot(urdf = 'urdfs/yumi_grippers.urdf',print_joint_info=True)
        # self.load_robot(urdf = 'environment/urdf/ur5_suction.urdf',print_joint_info=True)

        self._urdf = urdf
        # self._urdf = "ur5_suction"
        # self._urdf = "ur5_suction_big"
        
        
        self.load_robot(urdf = f'environment/urdf/{self._urdf}.urdf',print_joint_info=True)
        

        # self._FK_offset = np.array([0.0,-0.00,-0.02])
        self._FK_offset = np.array([0.0,-0.00,-0.0])
        
        self.reset_robot()        
        self._dummy_sim_step(1000)

        print("\n\n\nRobot is armed and ready to use...\n\n\n")
        print ('-'*40)
        
        camera_pos = np.array([-0.1, 0.0, 0.8])
        camera_target = np.array([0.6, 0, 0.])        
        self._init_camera(camera_pos,camera_target)
        

    

    def applyForce(self,object_id, force):
        p.applyExternalForce(object_id, -1, force, np.array([-0.0,-0.0,0]), p.LINK_FRAME)

    def move_figers(self, pos):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._FINGERS_JOINT_IDS,targetPositions  = [pos,pos])        
        self._dummy_sim_step(1)



    def fifth_order_trajectory_planner_3d(self,start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration):
        """
        Generates a fifth-order trajectory plan for 3D position considering velocity and acceleration,
        given the initial and final conditions.

        Args:
            start_pos (numpy.ndarray): Starting position as a 1D array of shape (3,) for (x, y, z).
            end_pos (numpy.ndarray): Ending position as a 1D array of shape (3,) for (x, y, z).
            start_vel (numpy.ndarray): Starting velocity as a 1D array of shape (3,) for (x, y, z).
            end_vel (numpy.ndarray): Ending velocity as a 1D array of shape (3,) for (x, y, z).
            start_acc (numpy.ndarray): Starting acceleration as a 1D array of shape (3,) for (x, y, z).
            end_acc (numpy.ndarray): Ending acceleration as a 1D array of shape (3,) for (x, y, z).
            duration (float): Desired duration of the trajectory.
            dt (float): Time step for the trajectory plan.

        Returns:
            tuple: A tuple containing time, position, velocity, and acceleration arrays for x, y, and z coordinates.
        """
        # Calculate the polynomial coefficients for each dimension
        t0 = 0.0
        t1 = duration
        t = np.arange(t0, t1, self._simulationStepTime)
        n = len(t)

        A = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                    [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                    [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                    [1, t1, t1 ** 2, t1 ** 3, t1 ** 4, t1 ** 5],
                    [0, 1, 2 * t1, 3 * t1 ** 2, 4 * t1 ** 3, 5 * t1 ** 4],
                    [0, 0, 2, 6 * t1, 12 * t1 ** 2, 20 * t1 ** 3]])

        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        acc = np.zeros((n, 3))

        for dim in range(3):
            b_pos = np.array([start_pos[dim], start_vel[dim], start_acc[dim], end_pos[dim], end_vel[dim], end_acc[dim]])
            x_pos = np.linalg.solve(A, b_pos)

            # Generate trajectory for the dimension using the polynomial coefficients
            pos[:, dim] = np.polyval(x_pos[::-1], t)
            vel[:, dim] = np.polyval(np.polyder(x_pos[::-1]), t)
            acc[:, dim] = np.polyval(np.polyder(np.polyder(x_pos[::-1])), t)

        return pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2], acc[:, 0], acc[:, 1],acc[:, 2]


    def fifth_order_trajectory_planner(self,start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration, dt):
        """
        Generates a fifth-order trajectory plan given the initial and final conditions.

        Args:
            start_pos (float): Starting position.
            end_pos (float): Ending position.
            start_vel (float): Starting velocity.
            end_vel (float): Ending velocity.
            start_acc (float): Starting acceleration.
            end_acc (float): Ending acceleration.
            duration (float): Desired duration of the trajectory.
            dt (float): Time step for the trajectory plan.

        Returns:
            tuple: A tuple containing time, position, velocity, and acceleration arrays.
        """
        # Calculate the polynomial coefficients
        t0 = 0.0
        t1 = duration
        t = np.arange(t0, t1, dt)
        n = len(t)

        A = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                    [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                    [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                    [1, t1, t1 ** 2, t1 ** 3, t1 ** 4, t1 ** 5],
                    [0, 1, 2 * t1, 3 * t1 ** 2, 4 * t1 ** 3, 5 * t1 ** 4],
                    [0, 0, 2, 6 * t1, 12 * t1 ** 2, 20 * t1 ** 3]])

        b_pos = np.array([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])

        x_pos = np.linalg.solve(A, b_pos)

        # Generate trajectory using the polynomial coefficients
        pos = np.polyval(x_pos[::-1], t)
        vel = np.polyval(np.polyder(x_pos[::-1]), t)
        acc = np.polyval(np.polyder(np.polyder(x_pos[::-1])), t)

        return t, pos, vel, acc
    
    def is_gripper_in_contact(self):
        if self._urdf == "ur5_suction":
            list_of_contacts = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIP_JOINT_ID[-1])
            if len (list_of_contacts)>0:
                return True
            else:
                return False
        elif self._urdf == "ur5_suction_big":
            list_of_contacts1 = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIPPER_SURF_ID[0])
            list_of_contacts2 = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIPPER_SURF_ID[1])
            list_of_contacts3 = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIPPER_SURF_ID[2])
            list_of_contacts4 = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIPPER_SURF_ID[3])
            list_of_contacts5 = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIPPER_SURF_ID[4])
            
            
            if len (list_of_contacts1)>0 or len (list_of_contacts2)>0 or \
               len (list_of_contacts3)>0 or len (list_of_contacts4)>0 or len (list_of_contacts5)>0:
                return True
            else:
                return False


    def suction_grasp(self,enable=True):
                
        if self._urdf == "ur5_suction":
            if enable:
                list_of_contacts = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIP_JOINT_ID[-1])            
                if len (list_of_contacts)>0:
                    obj_id = list_of_contacts[0][2]
                    list_of_contacts = p.getContactPoints(bodyA = self.robot_id, linkIndexA = self._GRIP_JOINT_ID[-1],bodyB =obj_id,linkIndexB= -1)

                    ee_pose = self.get_ee_state()
                    # contact_pos_A = np.array(list_of_contacts[0][5])
                    # contact_pos_B = np.array(list_of_contacts[0][6])
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    obj_dim  = p.getVisualShapeData(obj_id)[0][3]
                    obj_mass = p.getDynamicsInfo(obj_id,-1)[0]
                    p.changeDynamics(obj_id,-1, mass = 0.1)
                    ori = p.getEulerFromQuaternion(ee_pose[1])
                    ori = p.getQuaternionFromEuler(-np.array(ori))

                    self._suction_grasp = [p.createConstraint(
                                        self.robot_id,
                                        self._GRIP_JOINT_ID[-1],
                                        obj_id,
                                        -1,
                                        jointType=p.JOINT_FIXED,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0.05, 0, 0],
                                        # parentFrameOrientation= p.getQuaternionFromEuler([0,-np.pi/2,0]),
                                        parentFrameOrientation= ori,                                    
                                        childFramePosition=[0.0, 0, 0.],
                                        childFrameOrientation=[0,0,0,1])]
                    self._dummy_sim_step(10)
                    p.changeDynamics(obj_id, -1, mass = obj_mass)
                    return [obj_id]
                else:
                    self._suction_grasp = []
                    print("error: no object is in contact with the suction!")
                    return -1
                
            else:
                for s in self._suction_grasp:                
                    p.removeConstraint(s)
                self._suction_grasp = []
                self._dummy_sim_step(10)
                return 1000
            
        elif self._urdf == "ur5_suction_big":
            
            if enable:
                self._suction_grasp = []
                obj_list = []                
                gripper_in_use = []
                for surf_id in self._GRIPPER_SURF_ID:
                    list_of_contacts = p.getContactPoints(bodyA = self.robot_id, linkIndexA = surf_id)            
                    if len (list_of_contacts)>0:
                        obj_id = list_of_contacts[0][2]
                        obj_posA = list_of_contacts[0][5]
                        if obj_id in obj_list:
                            continue                   

                        list_of_contacts = p.getContactPoints(bodyA = self.robot_id, linkIndexA = surf_id,bodyB =obj_id,linkIndexB= -1)

                        ee_pose = self.get_ee_state()
                        # contact_pos_A = np.array(list_of_contacts[0][5])
                        # contact_pos_B = np.array(list_of_contacts[0][6])
                        obj_pose = p.getBasePositionAndOrientation(obj_id)
                        obj_dim  = p.getVisualShapeData(obj_id)[0][3]
                        obj_mass = p.getDynamicsInfo(obj_id,-1)[0]
                        p.changeDynamics(obj_id,-1, mass = 0.1)
                        ori = p.getEulerFromQuaternion(ee_pose[1])
                        # ori = p.getQuaternionFromEuler(-np.array(ori))
                        ori_obj = p.getQuaternionFromEuler([0,-np.pi/2,0]) if ori[1]>1 else p.getQuaternionFromEuler([0,0,0])

                        if surf_id in gripper_in_use:
                          continue


                        if surf_id == 11:
                            if obj_posA[1]>0 and not (10 in gripper_in_use):
                                attch_surf_id = 10 
                            elif not (12 in gripper_in_use):
                                attch_surf_id = 12
                            elif not (10 in gripper_in_use):
                                attch_surf_id = 10
                        
                        elif surf_id in [9,10]:
                            attch_surf_id = 10                            
                        elif surf_id in [12,13]:
                            attch_surf_id = 12
                        

                        

                        gripper_in_use.append(attch_surf_id)

                        self._suction_grasp.append (p.createConstraint(
                                            self.robot_id,
                                            attch_surf_id,
                                            obj_id,
                                            -1,
                                            jointType=p.JOINT_FIXED,
                                            jointAxis=[0, 0, 0],
                                            parentFramePosition=[0.05, 0, 0],
                                            # parentFrameOrientation= p.getQuaternionFromEuler([0,-np.pi/2,0]),
                                            # parentFrameOrientation= ori,    
                                            parentFrameOrientation= ori_obj,                                     
                                            childFramePosition=[0.0, 0, 0.],
                                            childFrameOrientation=[0,0,0,1]))
                        
                        obj_list.append(obj_id)       
                if obj_list == [] :
                    print("error: no object is in contact with the suction!")
                    return -1
                else:
                    return obj_list
                
            else:
                for s in self._suction_grasp:                
                    p.removeConstraint(s)
                self._suction_grasp = []
                self._dummy_sim_step(10)
                return 1000

            
        
                
    def move_object(self,grasp_object_id,duration):

        for i in range(int(duration/self._simulationStepTime)):
            # env.applyForce(grasp_object_id,[0,180,0])
            for obj_id in grasp_object_id:
                pos,ori = p.getBasePositionAndOrientation(obj_id)
                p.resetBasePositionAndOrientation(obj_id,pos+np.array([-0.0025,0,0]),ori)
            self._dummy_sim_step(1)


    def select_next_object(self,object_no):
        if self._urdf == "ur5_suction":            
            if object_no[1]+1 < self._col :
                object_no[1]+=1
            else:
                if object_no[2]+1 < self._depth:                    
                    object_no[2]+= 1
                    object_no[1] = 0
                else:
                    if object_no[0]+1 < self._row:
                        object_no[0]+=1
                        object_no[1] = 0                        
                        object_no[2] = 0
                        
                    else:
                        object_no = [0,0,0]
            return object_no
        elif self._urdf == "ur5_suction_big":

            if object_no[1]+2 < self._col :
                object_no[1]+=2
            else:
                if object_no[2]+1 < self._depth:                    
                    object_no[2]+= 1
                    object_no[1] = 0
                else:
                    if object_no[0]+1 < self._row:
                        object_no[0]+=1
                        object_no[1] = 0                        
                        object_no[2] = 0
                        
                    else:
                        object_no = [0,0,0]
            return object_no
        




    def spline_planner(self,start_pos,start_quat, goal_pos,goal_quat, duration):
        # Extract the position and orientation from the start and goal poses
        # start_pos, start_quat = start_pose[:3], start_pose[3:]
        # goal_pos, goal_quat = goal_pose[:3], goal_pose[3:]

        num_steps = int(duration/self._simulationStepTime)

        # Compute the time variable
        t = np.linspace(0, duration, num=num_steps)

        # Compute position trajectory using cubic spline interpolation
        pos_traj = np.zeros((3, len(t)))
        for i in range(3):
            pos_spline = CubicSpline([0, duration], [start_pos[i], goal_pos[i]])
            pos_traj[i] = pos_spline(t)

        # Compute orientation trajectory using spherical linear interpolation (slerp)

        # Generate trajectory for quaternion
        quaternions = []
        key_rots = Rotation.from_quat([start_quat, goal_quat])
        slerp = Slerp([0, duration], key_rots)
        for t in np.linspace(0, duration, num_steps):
            quaternion = slerp(t)  # Spherical linear interpolation for w component
            # quaternion = quaternion.normalized().as_quat()
            quaternion = quaternion.as_quat()
            
            quaternions.append(quaternion)


        # start_rot = Rotation.from_quat(start_quat)
        # goal_rot = Rotation.from_quat(goal_quat)
        # interp_rot = start_rot.interpolate(goal_rot, t/duration, 'slerp')
        # quat_traj = interp_rot.as_quat().T

        # Combine position and orientation trajectories into 6D pose trajectory
        # pose_traj = np.concatenate((pos_traj, np.array(quaternions).T), axis=0)

        return pos_traj.T, quaternions

    def polynomial_interpolation_6D(self,initial_pos,initial_ori, final_pos, final_ori, T):
        # Extract initial position and orientation
        p0 = np.array(initial_pos)
        q0 = Rotation.from_quat(initial_ori)
        # q0 = np.array(initial_ori)
        

        # Extract final position and orientation
        pf = np.array(final_pos)
        qf = Rotation.from_quat(np.array(final_ori))
        # qf = np.array(final_ori)
        

        # Calculate time interval
        dt = self._simulationStepTime  # Interval for interpolation
        num_steps = int(T / dt)

        # Calculate coefficients for position interpolation
        a0 = p0
        a1 = np.zeros(3)
        a2 = np.zeros(3)
        a3 = (pf - p0) / (2 * T**3)
        a4 = -(pf - p0) / (2 * T**4)
        a5 = (pf - p0) / (2 * T**5)

        # Generate trajectory for position
        positions = []
        for t in np.linspace(0, T, num_steps):
            position = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
            positions.append(position)

        # Generate trajectory for quaternion
        quaternions = []
        key_rots = Rotation.from_quat([initial_ori, final_ori])
        slerp = Slerp([0, T], key_rots)
        for t in np.linspace(0, T, num_steps):
            quaternion = slerp(t)  # Spherical linear interpolation for w component
            # quaternion = quaternion.normalized().as_quat()
            quaternion = quaternion.as_quat()
            
            quaternions.append(quaternion)

        # Return the interpolated trajectory
        trajectory = []
        for position, quaternion in zip(positions, quaternions):
            pose = np.concatenate((position, quaternion))
            trajectory.append(pose)

        return trajectory


    def load_robot (self,urdf, print_joint_info = False):        
        self.robot_id = p.loadURDF(urdf,[0, 0, 0.], p.getQuaternionFromEuler([0, 0, 0]),useFixedBase=True)
        
        # self.add_a_cube_without_collision(pos=[-0.0,-0.1,0.3],size=[0.2,0.2,0.6],color=[0.6,0.6,0.6,1])

        numJoints = p.getNumJoints(self.robot_id)
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        self._left_ee_frame_name  = 'ee_fixed_joint'
        
        # self._urdf = "ur5_suction"
        if self._urdf == "ur5_suction":            
            self._HOME_POSITION = [0, -1.544, 1.54, -1.54,-1.570, -1.570]
            self._JOINT_IDS = [0,1,2,3,4,5]            
        
        elif self._urdf == "ur5_suction_big":
            self._HOME_POSITION = [0, -1.544, 1.54, -1.54,-1.570, -1.570,0.0,0.0]
            self._JOINT_IDS = [0,1,2,3,4,5,10,12]
            self._FINGERS_JOINT_IDS = [10,12]
            self._GRIPPER_SURF_ID = [9,10,12,13,11]
        
        # self._HOME_POSITION = [0, -1.544, 1.54, -1.54,-1.570, -1.570]
        # self._JOINT_IDS = [0,1,2,3,4,5,10,12]
        # self._FINGERS_JOINT_IDS = [10,12]

        self._GRIP_JOINT_ID = [6,7,8]
        self._max_torques = [150, 150, 150, 28, 28, 28]


        self._jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable", "jointAxis", "parentFramePos", "parentFrameOrn"])
        
        self._joint_Damping =  [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 
                                0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
        for i in range(numJoints):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            jointAxis = info[13]
            parentFramePos = info[14]
            parentFrameOrn = info[15]
            controllable = True if jointName in controlJoints else False
            info = self._jointInfo(jointID, jointName, jointType, jointLowerLimit,
                            jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable,
                            jointAxis, parentFramePos, parentFrameOrn)

            if info.type == "REVOLUTE" or info.type == "PRISMATIC":# or True:  # set revolute joint to static
                p.setJointMotorControl2(self.robot_id, info.id, p.POSITION_CONTROL, targetPosition=0, force=500 )
                # p.setJointMotorControl2(self.robot_id, info.id, p.POSITION_CONTROL, targetPosition=0, force=jointMaxForce )
                
                if print_joint_info:
                    print (info)
                    print (jointType)                            
                    print ('-'*40)
                    
        p.changeVisualShape(self.robot_id, 0, rgbaColor=[0.68, 0.85, 0.90, 1])
        p.changeVisualShape(self.robot_id, 3, rgbaColor=[0.68, 0.85, 0.90, 1])
        p.changeVisualShape(self.robot_id, 5, rgbaColor=[0.68, 0.85, 0.90, 1])
        
          


    def _init_camera(self,camera_pos,camera_target,visulize_camera = False):        
        self._camera_pos = camera_pos
        self._camera_target = camera_target
        
        self.camera = Camera(cam_pos=self._camera_pos, cam_target= self._camera_target, near = 0.2, far = 2, size= [640, 480], fov=60)
        if visulize_camera:
            self.visualize_camera_position(self._camera_pos)

    def _init_in_hand_camera(self,camera_pos,camera_target,visulize_camera = False):        
        self._in_hand_camera_pos = camera_pos
        self._in_hand_camera_target = camera_target
        
        self.in_hand_camera = Camera(cam_pos=self._in_hand_camera_pos, cam_target= self._in_hand_camera_target, near = 0.2, far = 2, size= [640, 480], fov=60)
        if visulize_camera:
            self.visualize_camera_position(self._in_hand_camera_pos)

    def _dummy_sim_step(self,n):
        for _ in range(n):
            p.stepSimulation()
    
    def wait(self,sec):
        for _ in range(1+int(sec/self._simulationStepTime)):
            p.stepSimulation()

    def reset_robot(self):
        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL, jointIndices = self._JOINT_IDS,targetPositions  = self._HOME_POSITION)        
        self._dummy_sim_step(100)
    
    def go_home(self):
        self.move_arm (target_pos= np.array([0.4,0.,0.35]), target_ori=[0,np.pi/2,0],duration=5)
        # p0,o0 = self.get_ee_state()
        # start_pos = p0
        # end_pos = np.array([0.4,0.,0.5])
        # start_vel = np.array([.0, 0.0, 0.0])
        # end_vel = np.array([0.0, 0.0, .0])
        # start_acc = np.array([0.0, 0.0, 0.0])
        # end_acc = np.array([0.0, 0.0, 0.0])
        # duration = 5.0
        # xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
        #     start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration)
        
        # # ori = p.getQuaternionFromEuler([-np.pi/2,-np.pi/2, 0])        
        # # ori = p.getQuaternionFromEuler([-0,np.pi/2, 0])        
        # ori = p.getQuaternionFromEuler([-0,0, 0])        
        
        # for i in range(len(xl_pos)):
        #     xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
        #     pose = [xd,ori]      
        #     self._move_arm(traget_pose=pose)
        #     self._dummy_sim_step(1)

    def adjust_arm_pos(self,target_pos, target_ori):        
        pose = [target_pos,p.getQuaternionFromEuler(target_ori)]      
        self._move_arm(traget_pose=pose)
        self._dummy_sim_step(1)

    def move_arm(self,target_pos, target_ori,duration=0.001):
        p0,o0 = self.get_ee_state()
        pos, quat = self.spline_planner(p0,o0,target_pos,p.getQuaternionFromEuler(target_ori),duration)
        for i in range(len(pos)):
            pose = [pos[i],quat[i]]      
            self._move_arm(traget_pose=pose)
            self._dummy_sim_step(1)
        
        pose = [target_pos,p.getQuaternionFromEuler(target_ori)]      
        self._move_arm(traget_pose=pose)
        self._dummy_sim_step(10)
        

        # print(f"p0: [{p0[0]:4.4f}\t{p0[1]:4.4f}\t{p0[2]:4.4f}]")
        
        # start_pos = p0
        # end_pos = target_pos
        # start_vel = np.array([.0, 0.0, 0.0])
        # end_vel = np.array([0.0, 0.0, .0])
        # start_acc = np.array([0.0, 0.0, 0.0])
        # end_acc = np.array([0.0, 0.0, 0.0])
        
        # xl_pos, yl_pos, zl_pos, xl_vel, yl_vel, zl_vel, xl_acc, yl_acc, zl_acc = self.fifth_order_trajectory_planner_3d(
        #     start_pos, end_pos, start_vel, end_vel, start_acc, end_acc, duration)
        
        # ori = p.getQuaternionFromEuler(target_ori)        

        # for i in range(len(xl_pos)):
        #     xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
        #     pose = [xd,ori]      
        #     self._move_arm(traget_pose=pose)
        #     self._dummy_sim_step(1)

        # traj = self.polynomial_interpolation_6D(p0,o0,target_pos,p.getQuaternionFromEuler(target_ori) ,T = duration)
        # for pose in traj:
        #     # xd = np.array([xl_pos[i],yl_pos[i],zl_pos[i]])
        #     pose = [pose[0:3],pose[3:]]      
        #     self._move_arm(traget_pose=pose)
        #     self._dummy_sim_step(1)

        
                                
    def get_ee_state(self):
        pose = p.getLinkState(self.robot_id,self._GRIP_JOINT_ID[-1])[0:2]        
        return pose[0]+self._FK_offset , pose[1]
    

    def _move_arm(self,traget_pose):                
       
        # ll = np.array([-2.8,-2.4,-0.1,-2.1,-5.0,-1.25,-3.9])
        # ul = np.array([ 2.8, 0.7, 0.1, 1.3, 5.0, 2.25,3.9])
        # jr = np.array([ 5.6,-1.4, 0.1,-0.8, 10.0,3.5,4.8])
        # jd = np.array([ 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
        
        # joint_poses = p.calculateInverseKinematics(self.robot_id, 
        #                                            self._LEFT_HAND_JOINT_IDS[-1], 
        #                                            traget_pose[0], traget_pose[1],
        #                                            lowerLimits=ll,
        #                                            upperLimits=ul,
        #                                            jointRanges=jr)
        
        # current_joint_states = p.getJointStates(self.robot_id,self._JOINT_IDS)
        # current_joint_pose = [j[0] for j in current_joint_states]

        joint_poses = p.calculateInverseKinematics(self.robot_id, 
                                                   self._GRIP_JOINT_ID[-1], 
                                                   traget_pose[0], traget_pose[1],maxNumIterations = 1000)

        p.setJointMotorControlArray(self.robot_id,controlMode = p.POSITION_CONTROL,
                                    jointIndices = self._JOINT_IDS[0:6],
                                    targetPositions  = joint_poses[0:6])
    
  
    def add_a_cube(self,pos,size=[0.1,0.1,0.1],mass = 0.1, color = [1,1,0,1], textureUniqueId = None):

        # cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])
        # p.changeDynamics(obj_id, 
        #                 -1,
        #                 spinningFriction=800,
        #                 rollingFriction=0.0,
        #                 linearDamping=50.0)
        
        if textureUniqueId is not None:
            p.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)

        # cubesID.append(obj_id)
        
        p.stepSimulation()
        return obj_id 
    
    def add_a_cube_without_collision(self,pos,size=[0.1,0.1,0.1], color = [1,1,0,1]):

        # cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(0, box, vis, pos, [0,0,0,1])
        p.stepSimulation()
        return obj_id 
    
    def convert_pixel_to_robot_frame_manual(self,pos):
        pixel_meter_ratio  = 100 # 1 pixel = 1 cm 
        origin_pixel_coordinate = np.array([369, 233])

        diff = (pos - origin_pixel_coordinate) / pixel_meter_ratio        
        print (f"diff(mm): ({diff[0]},{diff[1]})")
        return diff




    def convert_pixel_to_robot_frame(self,pos):

        # Convert pixel coordinate to normalized image coordinates
        u_norm = pos[0] / self.camera.width
        v_norm = pos[1] / self.camera.height

        # Convert normalized image coordinates to camera coordinates
        u_cam = (2.0 * u_norm) - 1.0
        v_cam = 1.0 - (2.0 * v_norm)

        # Get camera projection matrix
        proj_matrix = np.array(self.camera.projection_matrix).reshape(4, 4) 

        # Convert camera coordinates to homogeneous coordinates
        camera_coords = np.array([u_cam, v_cam, -1.0, 1.0])

        # Apply projection matrix
        homogeneous_coords = proj_matrix @ camera_coords
        homogeneous_coords /= homogeneous_coords[3]  # Normalize by the fourth component

        # Get camera view matrix
        view_matrix = np.array(self.camera.view_matrix).reshape(4, 4) 

        # Extract rotation matrix from view matrix
        rotation_matrix = np.linalg.inv(view_matrix[:3, :3])

        # Extract translation vector from view matrix
        translation_vector = -rotation_matrix @ view_matrix[:3, 3]

        # Convert camera coordinates to robot frame coordinates
        robot_pos = rotation_matrix @ homogeneous_coords[:3] + translation_vector

        # Print the result
        print("Pixel Coordinate:", (pos[0], pos[1]))
        print("Robot Frame Coordinate:", tuple(robot_pos))

        return robot_pos

    def capture_image(self,removeBackground = False): 
        bgr, depth, _ = self.camera.get_cam_img()
        
        if (removeBackground):                      
           bgr = bgr-self.bgBGRBox+self.bgBGRWithoutBox
           depth = depth-self.bgDepthBox+self.bgDepthWithoutBox

        ##convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb,depth
    
    def in_hand_camera_capture_image(self):
        
        ee_pose = self.get_ee_state()
        trans_camera_pose = p.multiplyTransforms(ee_pose[0],ee_pose[1],[-0.2,0.0,0.1],[0,0,0,1])
        trans_target_pose = p.multiplyTransforms(ee_pose[0],ee_pose[1],[0.09,0.0,0.],[0,0,0,1])
        
        
        camera_pos = trans_camera_pose[0]
        camera_target = trans_target_pose[0]
        self._init_in_hand_camera(camera_pos,camera_target) 
        bgr, depth, _ = self.in_hand_camera.get_cam_img()
        
        ##convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb,depth
    

    def save_image(self,bgr):
        rgbim = Image.fromarray(bgr)
        # depim = Image.fromarray(bgr)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        rgbim.save('sam_harmony/InboxGraspPredicion_harmony/images/simImgs/rgbtest'+timestr+'.png')

    def random_color(self):
        rgb =  np.random.uniform(0.0,0.7,3)
        return np.concatenate((rgb,[1]))
    
    def add_box(self, box_centre,ori_offset = [0.0, 0.0, 0.]):
        id1 = p.loadURDF("environment_Yumi/urdf/objects/box.urdf",
                         box_centre,
                         p.getQuaternionFromEuler(ori_offset),
                         useFixedBase=True)
        

    def creat_pile_of_cube(self,number_of_cubes):
        obj_init_pos = [0.5, 0.4]
        cube_obj = []

        for i in range(number_of_cubes):
            r_x    = random.uniform(obj_init_pos[0] - 0.2, obj_init_pos[0] + 0.2)
            r_y    = random.uniform(obj_init_pos[1] - 0.2, obj_init_pos[1] + 0.2)
            roll   = random.uniform(0, np.pi)
            orn    = p.getQuaternionFromEuler([roll, 0, 0])
            pos    = [r_x, r_y, 0.15]
            obj_id = self.add_a_cube(pos=pos,size=[0.15,0.30,0.1],color=self.random_color())
            self._dummy_sim_step(50)
            cube_obj.append(obj_id)
            time.sleep(0.1)

        # self.obj_ids = self.tubeObj    
        self._dummy_sim_step(100)
        return cube_obj
    

    def distance_between_faces(self, idA, idB):
        """
        This function computes the distances between the faces of two boxes given their IDs.
        """
        positionA, orientationA = p.getBasePositionAndOrientation(idA)
        positionB, orientationB = p.getBasePositionAndOrientation(idB)
        sizeA = p.getVisualShapeData(idA)[0][3]  # Size of box A
        sizeB = p.getVisualShapeData(idB)[0][3]  # Size of box B

        rotation_matrix_A = np.array(p.getMatrixFromQuaternion(orientationA)).reshape(3, 3)
        rotation_matrix_B = np.array(p.getMatrixFromQuaternion(orientationB)).reshape(3, 3)

        facesA = np.array([[-sizeA[0]/2, 0, 0], [sizeA[0]/2, 0, 0], [0, -sizeA[1]/2, 0], [0, sizeA[1]/2, 0], [0, 0, -sizeA[2]/2], [0, 0, sizeA[2]/2]])
        facesB = np.array([[-sizeB[0]/2, 0, 0], [sizeB[0]/2, 0, 0], [0, -sizeB[1]/2, 0], [0, sizeB[1]/2, 0], [0, 0, -sizeB[2]/2], [0, 0, sizeB[2]/2]])

        facesA_world = np.dot(rotation_matrix_A, facesA.T).T + positionA
        facesB_world = np.dot(rotation_matrix_B, facesB.T).T + positionB

        distances = np.zeros(6)
        for i in range(6):
            distances[i] = np.min(np.linalg.norm(facesB_world - facesA_world[i], axis=1))

        return distances

    def find_best_box_to_pick(self,results):
        """
        This function finds the best box to pick and its face to grasp based on the specified rules.
        The results parameter is a 2D array where each row corresponds to a box. The first entry in each row is the box ID, and the next six entries indicate whether the corresponding face of the box is in contact with another box (1 for yes, 0 for no).
        """
        best_box_id = None
        best_face = None
        min_distance = np.inf

        for row in results:
            box_id = row[0]
            top_face_in_contact = row[6]
            front_face_in_contact = row[1]

            if top_face_in_contact == 1:  # Rule 5: If the top face is not free, the box is not graspable
                continue

            position, _ = p.getBasePositionAndOrientation(box_id)
            distance = np.linalg.norm(position)

            if top_face_in_contact == 0:  # Rule 3: If the top face is free, grasping from top is better than front
                face = 'top'
            elif front_face_in_contact == 0:  # Rule 1: The grasp face could be either top or front
                face = 'front'
            else:
                continue  # Rule 2: The selected grasp face should not be in contact with the others

            if distance < min_distance:  # Rule 4: A box with lower distance to origin is better to pick
                best_box_id = box_id
                best_face = face
                min_distance = distance

        if best_box_id is None:
            return None, None  # No box can be picked
        else:
            return best_box_id, best_face
        
    def create_structured_box(self,row, col, depth):
        obj_init_pos = [0.7, -0.2]
        cube_obj = []
        self.box_size = [0.08,0.12,0.08]
        box_mass = 25

        self._row = row
        self._col = col
        self._depth = depth


        self._rack_id = self.add_a_cube(pos=[0.8,0.07,0.0],mass=300, size=[0.4,0.8,0.15],color=[0,0,0,1])
        self._convey  = self.add_a_cube(pos=[-0.55,0.4,0.0],mass=300, size=[2.2,0.5,0.005],color=[0.3,0.3,0.3,1])

        self._dummy_sim_step(100)
        for k in range(depth):
            for i in range(row):
                for j in range(col):                
                    r_x    = obj_init_pos[0]+i*self.box_size[0] +0.02+ 1*random.uniform( -0.01, 0.01)
                    r_y    = obj_init_pos[1]+j*self.box_size[1] +0.02+ 1*random.uniform( -0.01, 0.01)
                    roll   = 0 #random.uniform(0, np.pi)
                    orn    = p.getQuaternionFromEuler([roll, 0, 0])
                    pos    = [r_x, r_y, 0.1+k*self.box_size[2]+0.2]
                    obj_id = self.add_a_cube(pos=pos,mass=box_mass, size=self.box_size,color=self.random_color())
                    self._dummy_sim_step(1)
                    cube_obj.append(obj_id)
                    # time.sleep(0.1)

        # self.obj_ids = self.tubeObj    
        self._dummy_sim_step(100)
        return cube_obj
    
