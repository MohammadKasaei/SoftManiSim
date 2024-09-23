# import pybullet as p
import pybullet_data
import numpy as np
import math

from visualizer.visualizer import ODE
from scipy.spatial.transform import Rotation as Rot
import cv2
from pybullet_env.camera.camera import Camera

class SoftRobotBasicEnvironment():
    def __init__(self,
                 bullet = None,
                 body_color = [0.5, .0, 0.6, 1], 
                 head_color = [0., 0, 0.75, 1],
                 body_sphere_radius=0.02,
                 number_of_sphere = 30,
                 number_of_segment=3,
                 gui=True) -> None:
        
        self._simulationStepTime = 0.005
        self.GUI = gui
        self._sphere_radius = body_sphere_radius 
        self._number_of_segment = number_of_segment 
        if bullet is None:
            import pybullet as p
            self.bullet = p
            self.bullet.connect(self.bullet.GUI if self.GUI else self.bullet.DIRECT)
            self.bullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.bullet.setGravity(0, 0, -9.81)
            self.bullet.setTimeStep(self._simulationStepTime)
            self.plane_id = p.loadURDF('plane.urdf')

            self.bullet.configureDebugVisualizer(self.bullet.COV_ENABLE_GUI, 0)
            self.bullet.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=180, cameraPitch=-35,
                                     cameraTargetPosition=[0., 0, 0.1])
        
        else:
            self.bullet = bullet
            
        self._marker_ID = None
        self._ode = ODE()
        
        self._max_grasp_width = 0.01
        self._grasp_width = 1* self._max_grasp_width
        self._eyeToHand_camera_enabled = True
        self._eyeInHand_camera_enabled = True
        
        self._number_of_sphere = number_of_sphere    
        self._body_color = body_color    
        self._head_color = head_color    
        
        self.create_robot()

    def _dummy_sim_step(self, n):
        for _ in range(n):
            self.bullet.stepSimulation()

    def add_a_cube_without_collision(self, pos, size=[0.1, 0.1, 0.1], color=[0.1, 0.1, 0.1, 1], textureUniqueId=None):
        # cubesID = []
        box = self.bullet.createCollisionShape(self.bullet.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2])
        vis = self.bullet.createVisualShape(self.bullet.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2], rgbaColor=color)
        obj_id = self.bullet.createMultiBody(0, box, vis, pos, [0, 0, 0, 1])
        self.bullet.stepSimulation()
        if textureUniqueId is not None:
            self.bullet.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)
        return obj_id

    def add_a_cube(self, pos, ori= [0,0,0,1], size=[0.1, 0.1, 0.1], mass=0.1, color=[1, 1, 0, 1], textureUniqueId=None):
        # cubesID = []
        box = self.bullet.createCollisionShape(self.bullet.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2])
        vis = self.bullet.createVisualShape(self.bullet.GEOM_BOX, halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2], rgbaColor=color)
        obj_id = self.bullet.createMultiBody(mass, box, vis, pos, ori)
        # self.bullet.changeDynamics(obj_id,
        #                  -1,
        #                  spinningFriction=800,
        #                  rollingFriction=0.0,
        #                  linearDamping=50.0)

        if textureUniqueId is not None:
            self.bullet.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)

        self.bullet.stepSimulation()
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
        
        if pitch < 0 : 
            pitch += np.pi*2
        if yaw < 0 : 
            yaw += np.pi*2
            

        return self.bullet.getQuaternionFromEuler([roll, pitch, yaw]),[roll, pitch, yaw]

    def is_robot_in_contact(self,obj_id):
        
        for body in self._robot_bodies:
            # Get AABBs of the objects
            aabb1 = self.bullet.getAABB(body)
            aabb2 = self.bullet.getAABB(obj_id)

            # Check for overlap
            overlap = (aabb1[0][0] <= aabb2[1][0] and aabb1[1][0] >= aabb2[0][0] and
                    aabb1[0][1] <= aabb2[1][1] and aabb1[1][1] >= aabb2[0][1] and
                    aabb1[0][2] <= aabb2[1][2] and aabb1[1][2] >= aabb2[0][2])
            
            if overlap:
                return True
            
        return False

        
    def is_tip_in_contact(self,obj_id):
        # list_of_contacts = self.bullet.getContactPoints(bodyA = self._robot_bodies[-3], linkIndexA = -1)
        # if len (list_of_contacts)>0:
        #     return True
        # else:
        #     return False
        
        aabb1 = self.bullet.getAABB(self._robot_bodies[-3])
        aabb2 = self.bullet.getAABB(obj_id)

        # Check for overlap
        return (aabb1[0][0] <= aabb2[1][0] and aabb1[1][0] >= aabb2[0][0] and
                aabb1[0][1] <= aabb2[1][1] and aabb1[1][1] >= aabb2[0][1] and
                aabb1[0][2] <= aabb2[1][2] and aabb1[1][2] >= aabb2[0][2])
        
    
    def apply_force(self,force_magnitude,obj_id):
        pos1, _ = self.bullet.getBasePositionAndOrientation(self._robot_bodies[-3])
        pos2, _ = self.bullet.getBasePositionAndOrientation(obj_id)
        direction = [pos2[i] - pos1[i] for i in range(3)]
        
        # Normalize the direction vector
        norm = sum(x**2 for x in direction) ** 0.5
        direction = [x / norm for x in direction]
        
        # force_magnitude = 1  # Adjust this value as needed
        force = [force_magnitude * x for x in direction]
        self._env.bullet.applyExternalForce(obj_id, -1, force, [0, 0, 0],  self.bullet.WORLD_FRAME)
        self.bullet.stepSimulation()

    def suction_grasp(self,enable=True):
        if enable:
            list_of_contacts = self.bullet.getContactPoints(bodyA = self._robot_bodies[-3], linkIndexA = -1)            
            if len (list_of_contacts)>0:
                obj_id = list_of_contacts[0][2]
                list_of_contacts = self.bullet.getContactPoints(bodyA = self._robot_bodies[-3], linkIndexA = -1,bodyB =obj_id,linkIndexB= -1)

                ee_pose = self._head_pose
                # contact_pos_A = np.array(list_of_contacts[0][5])
                # contact_pos_B = np.array(list_of_contacts[0][6])
                obj_pose = self.bullet.getBasePositionAndOrientation(obj_id)
                obj_dim  = self.bullet.getVisualShapeData(obj_id)[0][3]
                obj_mass = self.bullet.getDynamicsInfo(obj_id,-1)[0]
                self.bullet.changeDynamics(obj_id,-1, mass = 0)
                ori = self.bullet.getEulerFromQuaternion(ee_pose[1])
                # ori = self.bullet.getQuaternionFromEuler(-np.array(ori))

                self._suction_grasp = [self.bullet.createConstraint(
                                    self._robot_bodies[-3],
                                    -1,
                                    obj_id,
                                    -1,
                                    jointType=self.bullet.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0.01, 0, 0],
                                    # parentFrameOrientation= p.getQuaternionFromEuler([0,-np.pi/2,0]),
                                    parentFrameOrientation= ori,                                    
                                    childFramePosition=[0.0, 0, 0.],
                                    childFrameOrientation=[0,0,0,1])]
                self._dummy_sim_step(10)
                self.bullet.changeDynamics(obj_id, -1, mass = obj_mass)
                # self.bullet.changeDynamics(obj_id, -1, mass = 0)
                
                return [obj_id]
            else:
                self._suction_grasp = []
                print("error: no object is in contact with the suction!")
                return -1
            
    def create_robot(self):
        
        act = np.array([0, 0, 0])
        self._ode.updateAction(act)
        sol = self._ode.odeStepFull()

        self._base_pos_init = np.array([0, 0, 0.0])
        self._base_pos      = np.array([0, 0, 0.0])
        
        if self._eyeInHand_camera_enabled:
            camera_pos = np.array([0,0,0])
            camera_target = camera_pos+ self.rotate_point_3d([0.0, 0.1, -0.05],[0,0,0])
            self._init_in_hand_camera(camera_pos,camera_target) 
        
        # Define the shape and color parameters (change these as needed)
        radius = self._sphere_radius
        # self._number_of_sphere = number_of_sphere

        shape = self.bullet.createCollisionShape(self.bullet.GEOM_SPHERE, radius=radius)
        visualShapeId = self.bullet.createVisualShape(self.bullet.GEOM_SPHERE, radius=radius, rgbaColor=self._body_color)

        # visualShapeId_tip = self.bullet.createVisualShape(self.bullet.GEOM_BOX, halfExtents=[0.01, 0.002, 0.001], rgbaColor=[1, 0, 0, 1])
        visualShapeId_tip = self.bullet.createVisualShape(self.bullet.GEOM_BOX, halfExtents=[0.0, 0.002, 0.001], rgbaColor=[1, 0, 0, 1])
        visualShapeId_tip_ = self.bullet.createVisualShape(self.bullet.GEOM_SPHERE, radius=radius + 0.0025, rgbaColor=self._head_color)
        # visualShapeId_tip_ = self.bullet.createVisualShape(self.bullet.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=self._head_color)

        # Load the positions
        idx = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]

        # Create a body at each position
        self._robot_bodies = [self.bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                baseVisualShapeIndex=visualShapeId,
                                                basePosition=pos + self._base_pos) for pos in positions]

        ori, _ = self.calculate_orientation(positions[-2], positions[-1])
        self._robot_bodies.append(self.bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip_,
                                                    basePosition=positions[-1] + self._base_pos,
                                                    baseOrientation=ori))

        self._robot_bodies.append(self.bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                                    baseVisualShapeIndex=visualShapeId_tip,
                                                    basePosition=positions[-1] + self._base_pos+ [-0.01,0,0], baseOrientation=ori))
        
        self._robot_bodies.append(self.bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
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
    
        bgr, depth, _ = self.in_hand_camera.get_cam_img()
        
        ##convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb,depth
    
    
    def combine_euler_angles(self, euler_angles1, euler_angles2):
        from scipy.spatial.transform import Rotation as R

        # Convert Euler angles to rotation matrices
        r1 = R.from_euler('xyz', euler_angles1, degrees=False).as_matrix()
        r2 = R.from_euler('xyz', euler_angles2, degrees=False).as_matrix()
        
        # Combine the rotation matrices
        combined_rotation = np.dot(r1, r2)
        
        # Convert the combined rotation matrix to Euler angles
        combined_euler_angles = R.from_matrix(combined_rotation).as_euler('xyz', degrees=False)
        
        return combined_euler_angles

    
    def move_robot_ori(self,
                       action=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,]),
                       base_pos = np.array([0, 0, 0.]), 
                       base_orin = np.array([0,0,0]),
                       camera_marker=True):        
      
        if (np.shape(action)[0]<self._number_of_segment*3):
            action = np.concatenate((np.zeros((self._number_of_segment*3)-np.shape(action)[0]),action),axis=0) 
        
        self._ode._reset_y0()
        sol = None
        for n in range(self._number_of_segment):
            # self._ode._update_l0(self,l0)
            self._ode.updateAction(action[n*3:(n+1)*3])
            sol_n = self._ode.odeStepFull()
            self._ode.y0 = sol_n[:,-1]        
            
            if sol is None:
                sol = np.copy(sol_n)
            else:                
                sol = np.concatenate((sol,sol_n),axis=1)
            
        base_ori = self.bullet.getQuaternionFromEuler(base_orin)
        self._base_pos, _base_ori   = base_pos, base_ori 
        
        _base_pos_init = np.array(self.bullet.multiplyTransforms ([0,0,0], [0,0,0,1], self._base_pos_init, base_ori)[0])
        dp = self._base_pos - _base_pos_init        
                
            
        _base_pos_offset = np.array(self.bullet.multiplyTransforms ([0,0,0],[0,0,0,1],[0,-0.,0],base_ori)[0])

        idx = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
        self._robot_line_ids = []                    
        
        pose_in_word_frame = []
        for i, pos in enumerate(positions):
            pos, orin = self.bullet.multiplyTransforms (self._base_pos + _base_pos_offset, _base_ori, pos, [0,0,0,1])
            pose_in_word_frame.append(np.concatenate((np.array(pos),np.array(orin))))
            self.bullet.resetBasePositionAndOrientation(self._robot_bodies[i], pos , orin)
            

        head_pos = np.array(self.bullet.multiplyTransforms (self._base_pos+ _base_pos_offset, _base_ori, positions[-1] + np.array([0,0.,0]), [0,0,0,1])[0])
        
        _tip_ori, tip_ori_euler  = self.calculate_orientation(positions[-3], positions[-1]) # Pitch and roll are not correct
        _ , tip_ori = self.bullet.multiplyTransforms([0,0, 0], base_ori, [0,0,0], _tip_ori)
        
        
        gripper_pos1 = self.rotate_point_3d([0.02,-self._grasp_width, 0], tip_ori_euler)
        gripper_pos2 = self.rotate_point_3d([0.02,self._grasp_width, 0], tip_ori_euler)
        
        gripper_pos1 = np.array(self.bullet.multiplyTransforms (head_pos, _base_ori, gripper_pos1, [0,0,0,1])[0])
        gripper_pos2 = np.array(self.bullet.multiplyTransforms (head_pos, _base_ori, gripper_pos2, [0,0,0,1])[0])
        
        self.bullet.resetBasePositionAndOrientation(self._robot_bodies[-3], head_pos , base_ori)
        self._head_pose = [head_pos,base_ori]        
        self.bullet.resetBasePositionAndOrientation(self._robot_bodies[-2], gripper_pos1, tip_ori)
        self.bullet.resetBasePositionAndOrientation(self._robot_bodies[-1], gripper_pos2, tip_ori)
                
        if self._eyeInHand_camera_enabled:       
            object_pose = self.bullet.getBasePositionAndOrientation(self._robot_bodies[-4])
            
            cam_ori = np.array (self.bullet.getEulerFromQuaternion(tip_ori))
            cam_ori [0] = 0
            cam_ori = self.bullet.getQuaternionFromEuler(cam_ori)
            trans_target_pose = self.bullet.multiplyTransforms(object_pose[0],tip_ori,[0.1,0.0,-0.0],[0,0,0,1])
            camera_pose = self.bullet.multiplyTransforms(object_pose[0],tip_ori,[0.,0.0,-0.0],[0,0,0,1])
            if camera_marker:
                self._set_marker(np.array(trans_target_pose[0]))

            camera_target = np.array(trans_target_pose[0])
            self._init_in_hand_camera(camera_pose[0],camera_target) 
        
        self.bullet.stepSimulation()

        return pose_in_word_frame, sol #[:, -1]
    
    
    def calc_tip_pos(self,
                    action=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,]),
                    base_pos = np.array([0, 0, 0.]), 
                    base_orin = np.array([0,0,0])
                    ):        
      
        if (np.shape(action)[0]<self._number_of_segment*3):
            action = np.concatenate((np.zeros((self._number_of_segment*3)-np.shape(action)[0]),action),axis=0) 
        
        self._ode._reset_y0()
        sol = None
        for n in range(self._number_of_segment):
            # self._ode._update_l0(self,l0)
            self._ode.updateAction(action[n*3:(n+1)*3])
            sol_n = self._ode.odeStepFull()
            self._ode.y0 = sol_n[:,-1]        
            
            if sol is None:
                sol = np.copy(sol_n)
            else:                
                sol = np.concatenate((sol,sol_n),axis=1)
            
        base_ori = self.bullet.getQuaternionFromEuler(base_orin)
        self._base_pos, _base_ori   = base_pos, base_ori 
        
            
        _base_pos_offset = np.array(self.bullet.multiplyTransforms ([0,0,0],[0,0,0,1],[0,-0.,0],base_ori)[0])

        idx = np.linspace(0, sol.shape[1] - 1, self._number_of_sphere, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
        self._robot_line_ids = []                    
        
        pose_in_word_frame = []
        for i, pos in enumerate(positions):
            pos, orin = self.bullet.multiplyTransforms (self._base_pos + _base_pos_offset, _base_ori, pos, [0,0,0,1])
            pose_in_word_frame.append(np.concatenate((np.array(pos),np.array(orin))))
            self.bullet.resetBasePositionAndOrientation(self._robot_bodies[i], pos , orin)
            

        head_pose = self.bullet.multiplyTransforms (self._base_pos+ _base_pos_offset, _base_ori, positions[-1] + np.array([0,0.,0]), [0,0,0,1])
        return np.array(head_pose[0]),np.array(head_pose[1])
        
    
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
        
        
    def gripper_test(self,gt):
        if gt<10:
            self.set_grasp_width(gt/10.0)
        elif gt<20:
            self.set_grasp_width((20-gt)/10.0)
        elif gt<30:
            self.set_grasp_width((gt-20)/10.0)
        elif gt<40:
            self.set_grasp_width((40-gt)/10.0)
        
            
            
    def _set_marker(self,pos,ori = [0,0,0,1]):
        if self._marker_ID is None:
            marker_shape = self.bullet.createVisualShape(self.bullet.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0., 0.5])
            self._marker_ID = self.bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=marker_shape,
                                                    baseVisualShapeIndex=marker_shape,
                                                    basePosition= [pos[0],pos[1],pos[2]] , baseOrientation=ori)
        else:
            self.bullet.resetBasePositionAndOrientation(self._marker_ID, [pos[0],pos[1],pos[2]] , ori)
            
        self._dummy_sim_step(1)
         
    def wait(self, sec):
        for _ in range(1 + int(sec / self._simulationStepTime)):
            self.bullet.stepSimulation()

    def add_a_cube(self,pos,size=[0.1,0.1,0.1],mass = 0.1, color = [1,1,0,1], textureUniqueId = None):

        # cubesID = []
        box     = self.bullet.createCollisionShape(self.bullet.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = self.bullet.createVisualShape(self.bullet.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = self.bullet.createMultiBody(mass, box, vis, pos, [0,0,0,1])
        self.bullet.changeDynamics(obj_id, 
                        -1,
                        spinningFriction=800,
                        rollingFriction=0.0,
                        linearDamping=50.0)
        
        if textureUniqueId is not None:
            self.bullet.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)

        # cubesID.append(obj_id)
        
        self.bullet.stepSimulation()
        return obj_id 
    