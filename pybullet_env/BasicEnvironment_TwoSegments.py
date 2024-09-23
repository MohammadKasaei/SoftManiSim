import pybullet as p
import pybullet_data
import numpy as np
import math

from visualizer.visualizer import ODE
from scipy.spatial.transform import Rotation as Rot
import cv2
from pybullet_env.camera.camera import Camera

class SoftRobotBasicEnvironment_TwoSegments():
    def __init__(self,moving_base = False) -> None:
        self._simulationStepTime = 0.005
        self.vis = True

        p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self._simulationStepTime)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=180, cameraPitch=-35,
                                     cameraTargetPosition=[0., 0, 0.1])
        self._marker_ID = None
        self._pybullet = p
        self.plane_id = p.loadURDF('plane.urdf')
        self._ode = ODE()
        
        self._max_grasp_width = 0.01
        self._grasp_width = 1* self._max_grasp_width
        self._eyeToHand_camera_enabled = True
        self._eyeInHand_camera_enabled = True
        
        if moving_base == False:
            self._robot_type = 0
            self.create_robot()
        else:
            self._robot_type = 1
            self.create_mobile_robot()

    def _dummy_sim_step(self, n):
        for _ in range(n):
            p.stepSimulation()

    def add_a_cube_without_collision(self, pos, size=[0.1, 0.1, 0.1], color=[0.1, 0.1, 0.1, 1], textureUniqueId=None):
        # cubesID = []
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2], rgbaColor=color)
        obj_id = p.createMultiBody(0, box, vis, pos, [0, 0, 0, 1])
        p.stepSimulation()
        if textureUniqueId is not None:
            p.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)
        return obj_id

    def add_a_cube(self, pos, size=[0.1, 0.1, 0.1], mass=0.1, color=[1, 1, 0, 1], textureUniqueId=None):
        # cubesID = []
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2], rgbaColor=color)
        obj_id = p.createMultiBody(mass, box, vis, pos, [0, 0, 0, 1])
        p.changeDynamics(obj_id,
                         -1,
                         spinningFriction=800,
                         rollingFriction=0.0,
                         linearDamping=50.0)

        if textureUniqueId is not None:
            p.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)

        p.stepSimulation()
        return obj_id

    def calculate_orientation(self, point1, point2):
        # Calculate the difference vector
        diff = np.array(point2) - np.array(point1)

        # Calculate yaw (around z-axis)
        yaw = math.atan2(diff[1], diff[0])

        # Calculate pitch (around y-axis)
        pitch = math.atan2(-diff[2], math.sqrt(diff[0] ** 2 + diff[1] ** 2))

        # Roll is arbitrary in this context, setting it to zero
        roll = 0

        return p.getQuaternionFromEuler([roll, pitch, yaw]),[roll, pitch, yaw]

    def create_robot(self, number_of_sphere=20, body_base_color=[0.1, 0.2, 0.6, 1],
                     body_base_leg_color=[0.5, 0.5, 0.5, 1]):
        
        act = np.array([0, 0, 0])
        self._ode.updateAction(act)
        sol_1 = self._ode.odeStepFull()
        
        self._ode.y0 = sol_1[:,-1]
        self._ode.updateAction(act)    
        sol_2 = self._ode.odeStepFull()
        sol = np.concatenate((sol_1,sol_2),axis=1)

        self._base_pos_init = np.array([0, 0, 0.1])
        self._base_pos      = np.array([0, 0, 0.1])
        
        texUid = p.loadTexture("pybullet_env/textures/table_tecture.png")
        self.add_a_cube_without_collision(pos=[-0., 0., 0], size=[0.5, 0.5, 0.01], color=[0.7, 0.7, 0.7, 1],
                                          textureUniqueId=texUid)  # table

        self._body_id = self.add_a_cube_without_collision(pos=[0., -0.1, 0.1], size=[0.1, 0.2, 0.1], color=body_base_color)  # body
        self.add_a_cube_without_collision(pos=[0.041, -0.009, 0.05] , size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        self.add_a_cube_without_collision(pos=[-0.041, -0.009, 0.05], size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        self.add_a_cube_without_collision(pos=[0.041, -0.189, 0.05] , size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        self.add_a_cube_without_collision(pos=[-0.041, -0.189, 0.05], size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        
        
        self._camera_base_pos1 = np.array([-0.0, -0.005, 0.17])
        self._camera_base_pos2 = self._camera_base_pos1+[-0.0, -0.00, 0.03]
        self._camera_base_pos3 = self._camera_base_pos1+[-0.0,  0.005, 0.03]
        
        self._camera_id1 = self.add_a_cube_without_collision(pos=self._camera_base_pos1, size=[0.01, 0.01, 0.04], color=[0.6,0.6,0.6,1])       # camera
        self._camera_id2 = self.add_a_cube_without_collision(pos=self._camera_base_pos2, size=[0.04, 0.01, 0.02], color=[0.,0.6,0.6,1])   # camera
        self._camera_id3 = self.add_a_cube_without_collision(pos=self._camera_base_pos3, size=[0.02, 0.01, 0.01], color=[0.0,0.0,0,1])    # camera
        
        # self._set_marker([0,0.02,0.16])
        
        if self._eyeToHand_camera_enabled:
            camera_pos = np.array(p.getBasePositionAndOrientation(self._camera_id3)[0])
            camera_target = camera_pos+np.array([0.0, 0.1, -0.05]) 
            self._init_camera(camera_pos,camera_target)
        
        if self._eyeInHand_camera_enabled:
            camera_pos = np.array(p.getBasePositionAndOrientation(self._camera_id3)[0])
            camera_target = camera_pos+ self.rotate_point_3d([0.0, 0.1, -0.05],[0,0,0])
            self._init_in_hand_camera(camera_pos,camera_target) 
            
        
        # Define the shape and color parameters (change these as needed)
        radius = 0.01
        # self._number_of_sphere = number_of_sphere
        self._number_of_sphere = sol.shape[1]

        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        color =[0.6, .6, 0.6, 1]
        visualShapeId_1 = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        
        visualShapeId_tip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.002, 0.001], rgbaColor=[1, 0, 0, 1])
        visualShapeId_tip_ = p.createVisualShape(p.GEOM_SPHERE, radius=radius + 0.005, rgbaColor=[0., 0, 0.75, 1])

        # Load the positions
        idx = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
       
        # Create a body at each position
        self._robot_bodies = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                                baseVisualShapeIndex=visualShapeId_1,
                                                basePosition=pos + self._base_pos) for pos in positions]
        
        # self._robot_bodies_2 = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
        #                                         baseVisualShapeIndex=visualShapeId_2,
        #                                         basePosition=pos + self._base_pos) for pos in positions_2]
        
        # self._robot_bodies = self._robot_bodies_1 + self._robot_bodies_2

        ori , _ = self.calculate_orientation(positions[-2], positions[-1])
        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip_,
                                                    basePosition=positions[-1] + self._base_pos,
                                                    baseOrientation=ori))
        
        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip,
                                                    basePosition=positions[-1] + self._base_pos+ [-0.01,0,0], baseOrientation=ori))
        
        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip,
                                                    basePosition=positions[-1] + self._base_pos + [0.01,0,0], baseOrientation=ori))       
       

        self._robot_line_ids = []
        self._dummy_sim_step(1)
    
    def create_mobile_robot(self, number_of_sphere=20, color=[0.6, .6, 0.6, 1], body_base_color=[0.3, 0.3, 0.3, 1],
                     body_base_leg_color=[0.8, 0.8, 0.8, 1]):
        
        act = np.array([0, 0, 0])
        self._ode.updateAction(act)
        sol = self._ode.odeStepFull()

        self._base_pos_init = np.array([0, 0, 0.06])
        self._base_pos      = np.array([0, 0, 0.06])
        texUid = p.loadTexture("pybullet_env/textures/table_tecture.png")
        self.add_a_cube_without_collision(pos=[-0., 0., 0], size=[0.5, 0.5, 0.01], color=[0.7, 0.7, 0.7, 1],
                                          textureUniqueId=texUid)  # table

        self._body_id = self.add_a_cube_without_collision(pos=[0., -0.1, 0.06], size=[0.1, 0.2, 0.1], color=body_base_color)  # body
        self._v_rail  = self.add_a_cube_without_collision(pos=[0., 0.0, 0.005], size=[0.02, 0.4, 0.01], color=body_base_leg_color)  # rail
        self._h_rail  = self.add_a_cube_without_collision(pos=[0., -0.1, 0.005], size=[0.4, 0.02, 0.01], color=body_base_leg_color)  # rail
        
        
        # self.add_a_cube_without_collision(pos=[0.041, -0.009, 0.05] , size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        # self.add_a_cube_without_collision(pos=[-0.041, -0.009, 0.05], size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        # self.add_a_cube_without_collision(pos=[0.041, -0.189, 0.05] , size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        # self.add_a_cube_without_collision(pos=[-0.041, -0.189, 0.05], size=[0.02, 0.02, 0.1], color=body_base_leg_color) # legs
        
        self._camera_base_pos1 = np.array([-0.0, -0.005, 0.13])
        self._camera_base_pos2 = self._camera_base_pos1+[-0.0, -0.00, 0.03]
        self._camera_base_pos3 = self._camera_base_pos1+[-0.0,  0.005, 0.03]
        
        self._camera_id1 = self.add_a_cube_without_collision(pos=self._camera_base_pos1, size=[0.01, 0.01, 0.04], color=[0.6,0.6,0.6,1])  # camera
        self._camera_id2 = self.add_a_cube_without_collision(pos=self._camera_base_pos2, size=[0.04, 0.01, 0.02], color=[0.,0.6,0.6,1])   # camera
        self._camera_id3 = self.add_a_cube_without_collision(pos=self._camera_base_pos3, size=[0.02, 0.01, 0.01], color=[0.0,0.0,0,1])    # camera
        
        # camera_pos = np.array(p.getBasePositionAndOrientation(self._camera_id3)[0])
        # camera_target = camera_pos+np.array([0.0, 0.1, -0.05]) 
        # self._init_camera(camera_pos,camera_target)
        
        # self._set_marker([0,0.02,0.16])
        
        if self._eyeToHand_camera_enabled:
            camera_pos = np.array(p.getBasePositionAndOrientation(self._camera_id3)[0])
            camera_target = camera_pos+np.array([0.0, 0.1, -0.05]) 
            self._init_camera(camera_pos,camera_target)
        
        if self._eyeInHand_camera_enabled:
            camera_pos = np.array(p.getBasePositionAndOrientation(self._camera_id3)[0])
            camera_target = camera_pos+ self.rotate_point_3d([0.0, 0.1, -0.05],[0,0,0])
            self._init_in_hand_camera(camera_pos,camera_target) 
        
        # Define the shape and color parameters (change these as needed)
        radius = 0.01
        self._number_of_sphere = number_of_sphere

        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)

        visualShapeId_tip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.002, 0.001], rgbaColor=[1, 0, 0, 1])
        visualShapeId_tip_ = p.createVisualShape(p.GEOM_SPHERE, radius=radius + 0.005, rgbaColor=[0., 0, 0.75, 1])

        # Load the positions
        idx = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]

        # Create a body at each position
        self._robot_bodies = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                                baseVisualShapeIndex=visualShapeId,
                                                basePosition=pos + self._base_pos) for pos in positions]

        ori, _ = self.calculate_orientation(positions[-2], positions[-1])
        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip_,
                                                    basePosition=positions[-1] + self._base_pos,
                                                    baseOrientation=ori))

        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip,
                                                    basePosition=positions[-1] + self._base_pos+ [-0.01,0,0], baseOrientation=ori))
        
        self._robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip,
                                                    basePosition=positions[-1] + self._base_pos + [0.01,0,0], baseOrientation=ori))      
        
       

        self._robot_line_ids = []
        self._dummy_sim_step(1)
    
    
        
    def _init_camera(self,camera_pos,camera_target,visulize_camera = False):        
        self._camera_pos = camera_pos
        self._camera_target = camera_target
        
        self.camera = Camera(cam_pos=self._camera_pos, cam_target= self._camera_target, near = 0.01, far = 0.3, size= [640, 480], fov=60)
        if visulize_camera:
            self.visualize_camera_position(self._camera_pos)

    def _init_in_hand_camera(self,camera_pos,camera_target,visulize_camera = False):        
        self._in_hand_camera_pos = camera_pos
        self._in_hand_camera_target = camera_target
        
        self.in_hand_camera = Camera(cam_pos=self._in_hand_camera_pos, cam_target= self._in_hand_camera_target, near = 0.01, far = 0.3, size= [640, 480], fov=60)
        if visulize_camera:
            self.visualize_camera_position(self._in_hand_camera_pos)
                
    def capture_image(self,removeBackground = False): 
        if not self._eyeToHand_camera_enabled:
            return None, None
        bgr, depth, _ = self.camera.get_cam_img()       
        ##convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb,depth
        
    
    def in_hand_camera_capture_image(self):
        if not self._eyeInHand_camera_enabled:
            return None, None
        
        # ee_pose = self.get_ee_state()
        # trans_camera_pose = p.multiplyTransforms(ee_pose[0],ee_pose[1],[-0.2,0.0,0.1],[0,0,0,1])
        # trans_target_pose = p.multiplyTransforms(ee_pose[0],ee_pose[1],[0.09,0.0,0.],[0,0,0,1])
        
        
        # camera_pos = trans_camera_pose[0]
        # camera_target = trans_target_pose[0]
        # self._init_in_hand_camera(camera_pos,camera_target) 
        bgr, depth, _ = self.in_hand_camera.get_cam_img()
        
        ##convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb,depth
    
    
    def _move_robot_jac(self, action=np.array([0, 0, 0, 0, 0, 0]),base_pos = np.array([0, 0, 0.1])):
        self._ode._reset_y0()
        self._ode.updateAction(action[:3])
        sol_1 = self._ode.odeStepFull()
        
        self._ode.y0 = sol_1[:,-1]
        self._ode.updateAction(action[-3:])    
        sol_2 = self._ode.odeStepFull()
        sol = np.concatenate((sol_1,sol_2),axis=1)
        return sol[:3, -1]
        
    def move_robot(self, action=np.array([0, 0, 0, 0, 0, 0]),base_pos = np.array([0, 0, 0.1]), vis=True):        
        
        self._ode._reset_y0()
        self._ode.updateAction(action[:3])
        sol_1 = self._ode.odeStepFull()
        
        self._ode.y0 = sol_1[:,-1]
        self._ode.updateAction(action[-3:])    
        sol_2 = self._ode.odeStepFull()
        sol = np.concatenate((sol_1,sol_2),axis=1)
        
        self._base_pos = base_pos          
        dp = self._base_pos - self._base_pos_init
        if self._robot_type == 1: # moving robot
            r_pos = np.array ([0,-0.1+dp[1],0.005])
            p.resetBasePositionAndOrientation(self._h_rail, r_pos, (0, 0, 0, 1))
            

        p.resetBasePositionAndOrientation(self._body_id, self._base_pos+[0,-0.1,0], (0, 0, 0, 1))
        p.resetBasePositionAndOrientation(self._camera_id1, self._camera_base_pos1+dp, (0, 0, 0, 1))
        p.resetBasePositionAndOrientation(self._camera_id2, self._camera_base_pos2+dp, (0, 0, 0, 1))
        p.resetBasePositionAndOrientation(self._camera_id3, self._camera_base_pos3+dp, (0, 0, 0, 1))
        if self._eyeToHand_camera_enabled:
            camera_pos = np.array(p.getBasePositionAndOrientation(self._camera_id3)[0])
            camera_target = camera_pos+np.array([0.0, 0.1, -0.05]) 
            self._init_camera(camera_pos,camera_target) 
        
        
        
        idx = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
        self._robot_line_ids = []                    
        
        for i, pos in enumerate(positions):
            p.resetBasePositionAndOrientation(self._robot_bodies[i], pos + self._base_pos, (0, 0, 0, 1))

        tip_ori, tip_ori_euler  = self.calculate_orientation(positions[-2], positions[-1])
        gripper_pos1 = self.rotate_point_3d([0.02,-self._grasp_width,0],tip_ori_euler)
        gripper_pos2 = self.rotate_point_3d([0.02,self._grasp_width,0],tip_ori_euler)
        
        # p.resetBasePositionAndOrientation(self._robot_bodies[-3], positions[-1] + self._base_pos, tip_ori)        
        p.resetBasePositionAndOrientation(self._robot_bodies[-3], positions[-1] + self._base_pos,[0,0,0,1])                
        p.resetBasePositionAndOrientation(self._robot_bodies[-2], positions[-1] + self._base_pos + gripper_pos1 , tip_ori)
        p.resetBasePositionAndOrientation(self._robot_bodies[-1], positions[-1] + self._base_pos + gripper_pos2, tip_ori)
        p.stepSimulation()
        
        if self._eyeInHand_camera_enabled:
            
            ee_pose = p.getBasePositionAndOrientation(self._robot_bodies[-3])
            trans_camera_pose = p.multiplyTransforms(ee_pose[0],ee_pose[1],[0.,0.01,0.0],[0,0,0,1])
            trans_target_pose = p.multiplyTransforms(ee_pose[0],ee_pose[1],[0.,0.08,0.001],[0,0,0,1])
            camera_pos    = np.array(trans_camera_pose[0])
            camera_target = np.array(trans_target_pose[0])
            # print (camera_target-camera_pos)
            self._init_in_hand_camera(camera_pos,camera_target) 
        
        p.stepSimulation()

        return sol[:, -1]
    

    def rotate_point_3d(self, point, rotation_angles):
        """
        Rotates a 3D point around the X, Y, and Z axes.

        :param point: A tuple or list of 3 elements representing the (x, y, z) coordinates of the point.
        :param rotation_angles: A tuple or list of 3 elements representing the rotation angles (in rad) around the X, Y, and Z axes respectively.
        :return: A tuple representing the rotated point coordinates (x, y, z).
        """
        # Convert angles to radians
        # rotation_angles = np.radians(rotation_angles)
        
        rx, ry, rz = rotation_angles

        # Rotation matrices for X, Y, Z axes
        rotation_x = np.array([[1, 0, 0],
                            [0, np.cos(rx), -np.sin(rx)],
                            [0, np.sin(rx), np.cos(rx)]])
        
        rotation_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                            [0, 1, 0],
                            [-np.sin(ry), 0, np.cos(ry)]])
        
        rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                            [np.sin(rz), np.cos(rz), 0],
                            [0, 0, 1]])

        # Combined rotation matrix
        rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

        # Rotate the point
        rotated_point = np.dot(rotation_matrix, point)

        return tuple(rotated_point)


    def set_grasp_width (self,grasp_width_percent = 0):
        grasp_width_percent = 1 if grasp_width_percent>1 else grasp_width_percent         
        self._grasp_width = grasp_width_percent* self._max_grasp_width
        
    # def visulize(self, sol):
    #     idx = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere, dtype=int)
    #     positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
    #     self._robot_line_ids = []                    
        
    #     for i, pos in enumerate(positions):
    #         p.resetBasePositionAndOrientation(self._robot_bodies[i], pos + self._base_pos, (0, 0, 0, 1))

    #     tip_ori, tip_ori_euler  = self.calculate_orientation(positions[-2], positions[-1])
    #     gripper_pos1 = self.rotate_point_3d([0.02,-self._grasp_width,0],tip_ori_euler)
    #     gripper_pos2 = self.rotate_point_3d([0.02,self._grasp_width,0],tip_ori_euler)
        
    #     p.resetBasePositionAndOrientation(self._robot_bodies[-3], positions[-1] + self._base_pos, tip_ori)        
    #     p.resetBasePositionAndOrientation(self._robot_bodies[-2], positions[-1] + self._base_pos + gripper_pos1 , tip_ori)
    #     p.resetBasePositionAndOrientation(self._robot_bodies[-1], positions[-1] + self._base_pos + gripper_pos2, tip_ori)

    #     self._dummy_sim_step(10)
        
        
    def gripper_test(self,gt):
        if gt<10:
            self.set_grasp_width(gt/10.0)
        elif gt<20:
            self.set_grasp_width((20-gt)/10.0)
        elif gt<30:
            self.set_grasp_width((gt-20)/10.0)
        elif gt<40:
            self.set_grasp_width((40-gt)/10.0)
        
            
            
    def _set_marker(self,pos,radius = 0.01,color = [1, 0, 0., 0.5]):
        if self._marker_ID is None:
            marker_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            self._marker_ID = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=marker_shape,
                                                    baseVisualShapeIndex=marker_shape,
                                                    basePosition= [pos[0],pos[2],pos[1]] + self._base_pos, baseOrientation=(0, 0, 0, 1))
        else:
            p.resetBasePositionAndOrientation(self._marker_ID, [pos[0],pos[2],pos[1]] + self._base_pos, (0, 0, 0, 1))
        self._dummy_sim_step(10)
         
    def wait(self, sec):
        for _ in range(1 + int(sec / self._simulationStepTime)):
            p.stepSimulation()
