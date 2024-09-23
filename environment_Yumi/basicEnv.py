from environment.utilities import setupUR5,setupPanda
from environment.camera.camera import Camera, CameraIntrinsic
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import random
import cv2
from utils.Matrix_4x4 import Matrix_4x4
from utils.ycbObjectLoader import YcbObjects


class BaiscEnvironment:
    OBJECT_INIT_HEIGHT = 1.05
    GRIPPER_MOVING_HEIGHT = 1.25
    GRIPPER_GRASPED_LIFT_HEIGHT = 1.4
    TARGET_ZONE_POS = [0.5, 0.0, 0.685]
    SIMULATION_STEP_DELAY = 0.0005 #speed of simulator - the lower the fatser
    # FINGER_LENGTH = 0.04 #0.06
    Z_TABLE_TOP = 0.785
    GRIP_REDUCTION = 0.3

    def __init__(self, GUI=False, debug=False, robotType ="UR5", gripper_type='140',img_size = 220,simulationStepTime=0.01) -> None:
        self.vis = GUI
        self.debug = debug

        self.robotType = robotType #"Panda" #UR5

        # self.camPos = [0.05, -0.52, 1.3] # box size 0.35
        self.camPos = [0.05, -0.52, 1.5] # box size 0.35
        # self.camPos = np.array([0.05, -0.52, 1.0])
        self.camTarget = np.array([self.camPos[0], self.camPos[1], 0.785])
        IMG_SIZE = img_size
        self.camera = Camera(cam_pos=self.camPos, cam_target= self.camTarget, near = 0.2, far = 2, size= [IMG_SIZE, IMG_SIZE], fov=40)



        self.obj_init_pos = (self.camera.x, self.camera.y)
        self.obj_ids = []
        self.obj_positions = []
        self.obj_orientations = []  

        self_collisions = True
        if self_collisions:
            flags = (p.URDF_USE_SELF_COLLISION + p.URDF_USE_INERTIA_FROM_FILE)
        else:
            flags = (p.URDF_USE_INERTIA_FROM_FILE)


        # gripper_type='85'
        if gripper_type not in ('85', '140'):
            raise NotImplementedError(
                'Gripper %s not implemented.' % gripper_type)
        self.gripper_type = gripper_type
        
        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(simulationStepTime)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera( cameraDistance=0.8, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,-0.5,1])

        # self.visualizeCameraPosition()

        self.planeID = p.loadURDF('plane.urdf')
        self.tableID = p.loadURDF('environment/urdf/objects/table.urdf',
                                  [0.0, -0.65, 0.76],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True)
        self.target_table_id = p.loadURDF('environment/urdf/objects/target_table.urdf',
                                          [0.7, 0.0, 0.66],
                                          p.getQuaternionFromEuler([0, 0, 0]),
                                          useFixedBase=True)
        self.target_id = p.loadURDF('environment/urdf/objects/traybox.urdf',
                                    self.TARGET_ZONE_POS,
                                    p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True,
                                    globalScaling=0.7)
        self.UR5Stand_id = p.loadURDF('environment/urdf/objects/ur5_stand.urdf',
                                      [-0.7, -0.36, 0.0],
                                      p.getQuaternionFromEuler([0, 0, 0]),
                                      useFixedBase=True)

        if (self.robotType == "UR5"):
            self.robot_id = p.loadURDF('environment/urdf/ur5_robotiq_%s.urdf' % gripper_type,
                                   [0, 0, 0.0],  # StartPosition
                                   p.getQuaternionFromEuler([0, 0, 0]),  # StartOrientation
                                   useFixedBase=True,
                                   flags=flags)
                              
            self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
                setupUR5(p, self.robot_id, gripper_type)
            
            self.eef_id = 7  # ee_link 
            self.ee_id  = 7
            self.f1_id  = 13 #12
            self.f2_id  = 17 #17
            self.finger_length = 0.25 # should be changed based on gripper


            # Add force sensors
            p.enableJointForceTorqueSensor(self.robot_id, self.joints['left_inner_finger_pad_joint'].id)
            p.enableJointForceTorqueSensor(self.robot_id, self.joints['right_inner_finger_pad_joint'].id)

            # Change the friction of the gripper
            p.changeDynamics(self.robot_id, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=1)
            p.changeDynamics(self.robot_id, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=1)
      
            # Setup some Limit
            if (gripper_type == "85"):
                self.gripper_open_limit = (0.0, 0.1)
            else:
                self.gripper_open_limit = (0.2, 0.8)
                
            self.ee_position_limit = ((-0.8, 0.8),
                                    (-0.8, 0.8),
                                    (0.785, 1.4))
            
        else:
            self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                   [0, 0, 0.76],  # StartPosition
                                   p.getQuaternionFromEuler([0, 0, 0]),  # StartOrientation
                                   useFixedBase=True,
                                   flags=flags)
            self.joints, self.controlGripper, self.controlJoints =\
            setupPanda(p, self.robot_id, gripper_type)
            self.eef_id = 11 #8 # ee_link
            self.finger_length = -0.025

            # Add force sensors
            p.enableJointForceTorqueSensor(self.robot_id, self.joints['panda_finger_joint1'].id)
            p.enableJointForceTorqueSensor(self.robot_id, self.joints['panda_finger_joint2'].id)

            # Change the friction of the gripper
            p.changeDynamics(self.robot_id, self.joints['panda_finger_joint1'].id, lateralFriction=1)
            p.changeDynamics(self.robot_id, self.joints['panda_finger_joint2'].id, lateralFriction=1)

            # Setup some Limit
            self.gripper_open_limit = (0.0, 0.1)
            self.ee_position_limit = ((-0.8, 0.8),
                                    (-0.8, 0.8),
                                    (0.73, 1.4))
      




        self.resetRobot(gripper_type)
        self.ee_pp   = p.getLinkState(self.robot_id,self.eef_id)[0]
        self.ee_orn  = p.getEulerFromQuaternion(p.getLinkState(self.robot_id,self.eef_id)[1])   
        # self.cameraPos = p.multiplyTransforms(self.ee_pp,p.getLinkState(self.robot_id,self.eef_id)[1],[0,0,0.1],p.getQuaternionFromEuler([0,0,0]))
        self.updateBackgroundImage(0)

        self.objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])
        
    

    def addACube(self,pos,size,mass = 0.1, color = [1,1,0,1]):
     
        self.cubesID = []
        
        
        box  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])
        p.changeDynamics(obj_id, 
                        -1,
                        spinningFriction=0.001,
                        rollingFriction=0.001,
                        linearDamping=0.0)


        self.cubesID.append(obj_id)
        p.stepSimulation()

    def moveObject(self,obj_id,pos,orn):
        p.resetBasePositionAndOrientation(obj_id, pos, orn)
        p.stepSimulation()

    def getObjState(self,):
        pos, orn = p.getBasePositionAndOrientation(self.cubesID[0])
        return pos,orn
    
    def drawCircle(self,center, radius,color = [0,1,1], thickness = 5):
        nPoint = 25
        lineIDs = []
        for i in range(nPoint+1):
            p1 = [ center[0]+radius*np.sin(i*2*np.pi/nPoint), center[1]+radius*np.cos(i*2*np.pi/nPoint), center[2]]
            j  = i+1
            p2 = [center[0]+radius*np.sin(j*2*np.pi/nPoint), center[1]+radius*np.cos(j*2*np.pi/nPoint), center[2]]
            lineIDs.append (p.addUserDebugLine(p1,p2, color , thickness))
            # p.addUserDebugLine([center[0],center[1],0], [center[0]+radius*np.sin(i*2*np.pi/10),center[1]+radius*np.cos(i*2*np.pi/10),0], [1,0,1], 5)
        return lineIDs


    def drawRotatedRectangle(self,center=np.array((0.1,0.1,0.5)),dx=0.05,dy=0.05,theta=0.1):
        xtmp = np.copy(center)
        A = xtmp+np.array((-dx*np.cos(theta)-dy*np.sin(theta),-dx*np.sin(theta)+dy*np.cos(theta),0))
        B = xtmp+np.array((dx*np.cos(theta)-dy*np.sin(theta),dx*np.sin(theta)+dy*np.cos(theta),0))
        C = xtmp+np.array((dx*np.cos(theta)+dy*np.sin(theta),dx*np.sin(theta)-dy*np.cos(theta),0))
        D = xtmp+np.array((-dx*np.cos(theta)+dy*np.sin(theta),-dx*np.sin(theta)-dy*np.cos(theta),0))
        TargetRectLines = self.drawRectangle(A,B,C,D,color=[0,0,1])
        return TargetRectLines
        
    def drawRectangle(self,A, B,C,D,color = [0,1,1], thickness = 5):
        lineIDs = []
        lineIDs.append (p.addUserDebugLine(A,B, color , thickness))
        lineIDs.append (p.addUserDebugLine(B,C, color , thickness))
        lineIDs.append (p.addUserDebugLine(C,D, color , thickness))
        lineIDs.append (p.addUserDebugLine(D,A, color , thickness))
        return lineIDs
    
    # def drawRectangle(self,origin, w,h,color = [0,1,1], thickness = 5):
    #     A = origin
    #     B = A + np.array([w,0,0])
    #     C = B + np.array([0,h,0])
    #     D = C - np.array([w,0,0])

    #     p.addUserDebugLine(A,B , color , thickness)
    #     p.addUserDebugLine(B,C, color , thickness)
    #     p.addUserDebugLine(C,D, color , thickness)
    #     p.addUserDebugLine(D,A, color , thickness)
            
    # def isOnTable(self,pos):
    #     if (pos[0] >= self.tableOrigin[0] and pos[0] <= self.tableOrigin[0]+self.tableWidth and \
    #         pos[1] >= self.tableOrigin[1] and pos[1] <= self.tableOrigin[1]+self.tableHeight):
    #         return True
    #     else:
    #         return False



    def stepSimulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
       
            
    def visualizeCameraPosition(self):
        pos = np.copy(self.camPos[:])+np.array([0,0,0.0025])
        size = 0.05
        halfsize = size/2
        mass  = 0 #kg
        color = [1,0,0,1]
        lens  = p.createCollisionShape(p.GEOM_CYLINDER, radius=halfsize*2,height = 0.005 )
        vis = p.createVisualShape(p.GEOM_CYLINDER,radius=halfsize*2,length=0.005, rgbaColor=color, specularColor=[0,0,0])
        obj_id = p.createMultiBody(mass, lens, vis, pos, [0,0,0,1])
        
        ####

        color = [0,0,0,1]
        pos = np.copy(self.camPos[:])+np.array([0,0,0.025+0.0025])
        box  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[halfsize, halfsize, halfsize])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[halfsize, halfsize, halfsize], rgbaColor=color, specularColor=[1,1,1])
        obj_id = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])

        


        # p.changeDynamics(obj_id, 
        #                 -1,
        #                 spinningFriction=0.001,
        #                 rollingFriction=0.001,
        #                 linearDamping=0.0)

    def visualizePredictedGrasp(self,grasps,color = [0,0,1],visibleTime =2):
       
        lineIDs = []
        for g in grasps:
            x, y, z, yaw, opening_len, obj_height = g
            opening_len = np.clip(opening_len,0,0.04)
            yaw = yaw-np.pi/2
            self.lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=5))
            self.lineIDs.append(p.addUserDebugLine([x, y, z], [x+(opening_len*np.cos(yaw)), y+(opening_len*np.sin(yaw)), z],color, lineWidth=5))
            self.lineIDs.append(p.addUserDebugLine([x, y, z], [x-(opening_len*np.cos(yaw)), y-(opening_len*np.sin(yaw)), z],color, lineWidth=5))
            

        self.dummySimulationSteps(10)
        time.sleep(visibleTime)
        self.removeDrawing(self.lineIDs)

    def removeDrawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)
    
    def dummySimulationSteps(self,n):
        for _ in range(n):
            p.stepSimulation()
    
                
    def isThereAnyObject(self):
        self.dummySimulationSteps(10)
        rgb, depth = self.captureImage(1)

        #print ("min RGB = ", rgb.min(), "max RGB = ", rgb.max(), "rgb.avg() = ", np.average(rgb))
        # print ("min depth = ", depth.min(), "max depth = ", depth.max())
        if (depth.max()- depth.min() < 0.005):
            return False
        else:
            return True   
    
    def updateBackgroundImage(self,stage):
        self.dummySimulationSteps(100)

        if (stage == 0):
            self.bgBGRWithoutBox, self.bgDepthWithoutBox, _ = self.camera.get_cam_img()
        elif (stage == 1):
            self.bgBGRBox, self.bgDepthBox, _ = self.camera.get_cam_img()
            

    def captureImage(self,removeBackground=0): 
        bgr, depth, _ = self.camera.get_cam_img()
        
        if (removeBackground):                      
           bgr = bgr-self.bgBGRBox+self.bgBGRWithoutBox
           depth = depth-self.bgDepthBox+self.bgDepthWithoutBox

        ##convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb,depth


    @staticmethod
    def isStable(handle,still_eps = 1e-3):
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def waittingToBeStable(self, objID, max_wait_epochs=10):
        for _ in range(max_wait_epochs):
            self.stepSimulation()
            if self.isStable(objID):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def waittingForAllToBeStable(self, max_wait_epochs=500):
        for _ in range(max_wait_epochs):
            self.stepSimulation()
            if np.all(list(self.isStable(obj_id) for obj_id in self.obj_ids)):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def resetRobot(self,gripperType):

        if self.robotType == "UR5":
            # user_parameters = (0, -1.544, 1.54, -1.54,-1.570, 0.000, 0.085)
            user_parameters = (0, -1.544, 1.54, -1.54,-1.570, 0.000, (0.085 if gripperType == "85" else 0.3))
            
        
        elif self.robotType == "Panda":
            user_parameters = (0.98, 0.458, 0.31, -2.24, -0.30+np.pi/2, 2.66, 2.32, 0.02, 0.02)
        
        else:
            user_parameters = (0.98, 0.458, 0.31, -2.24, -0.30+np.pi/2, 2.66, 2.32, 0.02, 0.02)
      
        for _ in range(60):
            for i, name in enumerate(self.controlJoints):
                
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                self.stepSimulation()
        
                    
        self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=(0.085 if gripperType == "85" else 0.5))
        self.stepSimulation()
        


 
    def goHome(self):
        y_orn = p.getQuaternionFromEuler([-np.pi*0.25, np.pi/2, 0.0])
        self.moveEE(np.array((self.TARGET_ZONE_POS[0],
                     self.TARGET_ZONE_POS[1], 1.25)), y_orn,max_step= 1000)


    def moveArmAway(self):
        joint = self.joints['shoulder_pan_joint']
        for _ in range(200):
            p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                    targetPosition=0., force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
            self.stepSimulation()

    def checkGrasped(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left +
                          contact_right if item[2] in [self.obj_id])
        if len(contact_ids) == 1:
            return True
        return False

    def checkGraspedID(self):
        if (self.robotType == "UR5"):
            left_index = self.joints['left_inner_finger_pad_joint'].id
            right_index = self.joints['right_inner_finger_pad_joint'].id
        else:
            left_index = self.joints['panda_finger_joint1'].id
            right_index = self.joints['panda_finger_joint2'].id
     
        contact_left = p.getContactPoints(bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left +contact_right if item[2] in self.obj_ids)
        if len(contact_ids) > 1:
            if self.debug:
                print('Warning: Multiple items in hand!')
        return list(item_id for item_id in contact_ids if item_id in self.obj_ids)

    def checkContact(self, id_a, id_b):
        contact_a = p.getContactPoints(bodyA=id_a)
        contact_ids = set(item[2] for item in contact_a if item[2] in [id_b])
        if len(contact_ids) == 1:
            return True
        return False

    def IsTargetReached(self, obj_id):
        aabb = p.getAABB(self.target_id, -1)
        x_min, x_max = aabb[0][0], aabb[1][0]
        y_min, y_max = aabb[0][1], aabb[1][1]
        pos = p.getBasePositionAndOrientation(obj_id)
        x, y = pos[0][0], pos[0][1]
        if x > x_min and x < x_max and y > y_min and y < y_max:
            return True
        return False

    def GripperContact(self, bool_operator='and', force=100):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(self.robot_id, left_index)[
            2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robot_id, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(
            left_force), np.linalg.norm(right_force)
        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

    def moveGripper(self, gripper_opening_length: float, step: int = 1):

        
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        # gripper_opening_angle = 0.715 - np.clip()  # angle calculation
    
        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL,targetPosition=gripper_opening_length)

            self.stepSimulation()

    def calcZOffset(self, gripper_opening_length: float):
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)
        if self.gripper_type == '140':
            gripper_length = 10.3613 * np.sin(1.64534-0.24074 * (gripper_opening_angle / np.pi)) - 10.1219
        else:
            gripper_length = 1.231 - 1.1
        return gripper_length

    def removeObject(self, obj_id):
        # Get index of obj in id list, then remove object from simulation
        idx = self.obj_ids.index(obj_id)
        self.obj_orientations.pop(idx)
        self.obj_positions.pop(idx)
        self.obj_ids.pop(idx)
        p.removeBody(obj_id)

    def removeAllObject(self):
        self.obj_positions.clear()
        self.obj_orientations.clear()
        for obj_id in self.obj_ids:
            p.removeBody(obj_id)
        self.obj_ids.clear()

    def reset_all_obj(self):
        for i, obj_id in enumerate(self.obj_ids):
            p.resetBasePositionAndOrientation(
                obj_id, self.obj_positions[i], self.obj_orientations[i])
        self.waittingForAllToBeStable()

    def updateObjectStates(self):
        for i, obj_id in enumerate(self.obj_ids):
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            self.obj_positions[i] = pos
            self.obj_orientations[i] = orn

    def loadObj(self, path, pos, yaw, mod_orn=False, mod_stiffness=False):
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        obj_id = p.loadURDF(path, pos, orn)
        # adjust position according to height
        aabb = p.getAABB(obj_id, -1)
        if mod_orn:
            minm, maxm = aabb[0][1], aabb[1][1]
            orn = p.getQuaternionFromEuler([0, np.pi*0.5, yaw])
        else:
            minm, maxm = aabb[0][2], aabb[1][2]

        pos[2] += (maxm - minm) / 2
        p.resetBasePositionAndOrientation(obj_id, pos, orn)
        # change dynamics
        if mod_stiffness:
            p.changeDynamics(obj_id,
                             -1, lateralFriction=1,
                             rollingFriction=0.001,
                             spinningFriction=0.002,
                             restitution=0.01,
                             contactStiffness=100000,
                             contactDamping=0.0)
        else:
            p.changeDynamics(obj_id,
                             -1, lateralFriction=1,
                             rollingFriction=0.002,
                             spinningFriction=0.001,
                             restitution=0.01)
        self.obj_ids.append(obj_id)
        self.obj_positions.append(pos)
        self.obj_orientations.append(orn)
        return obj_id, pos, orn

    def loadIsolatedObj(self, path, mod_orn=False, mod_stiffness=False):
        r_x = random.uniform(
            self.obj_init_pos[0] - 0.1, self.obj_init_pos[0] + 0.1)
        r_y = random.uniform(
            self.obj_init_pos[1] - 0.1, self.obj_init_pos[1] + 0.1)
        yaw = random.uniform(0, np.pi)

        pos = [r_x, r_y, self.Z_TABLE_TOP]
        obj_id, _, _ = self.loadObj(path, pos, yaw, mod_orn, mod_stiffness)
        for _ in range(10):
            self.stepSimulation()
        self.waittingToBeStable(obj_id)
        self.updateObjectStates()
        for _ in range(200):
            p.stepSimulation()

        return obj_id


            
    def createTempBox(self, width, no):
        box_width = width
        box_height = 0.1
        box_z = self.Z_TABLE_TOP + (box_height/2)
        id1 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [self.obj_init_pos[0] - box_width /
                             2, self.obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id2 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [self.obj_init_pos[0] + box_width /
                             2, self.obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id3 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [self.obj_init_pos[0], self.obj_init_pos[1] +
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        id4 = p.loadURDF(f'environment/urdf/objects/slab{no}.urdf',
                         [self.obj_init_pos[0], self.obj_init_pos[1] -
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        self.updateBackgroundImage(1)
                 
        return [id1, id2, id3, id4]

    def createPile(self, obj_info):
        box_ids = self.createTempBox(0.35, 1)
        for path, mod_orn, mod_stiffness in obj_info:
            margin = 0.025
            r_x = random.uniform(
                self.obj_init_pos[0] - margin, self.obj_init_pos[0] + margin)
            r_y = random.uniform(
                self.obj_init_pos[1] - margin, self.obj_init_pos[1] + margin)
            yaw = random.uniform(0, np.pi)
            pos = [r_x, r_y, 1.0]

            obj_id, _, _ = self.loadObj(
                path, pos, yaw, mod_orn, mod_stiffness)
            for _ in range(10):
                self.stepSimulation()
            self.waittingToBeStable(obj_id, 30)

        self.waittingForAllToBeStable()
        for handle in box_ids:
            p.removeBody(handle)
        box_ids = self.createTempBox(0.45, 2)
        self.waittingForAllToBeStable(100)
        for handle in box_ids:
            p.removeBody(handle)
        self.waittingForAllToBeStable()
        self.updateObjectStates()

    def moveObjectAlongAxis(self, obj_id, axis, operator, step, stop):
        collison = False
        while not collison:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            new_pos = list(pos)
            if operator == '+':
                new_pos[axis] += step
                if new_pos[axis] > stop:
                    break
            else:
                new_pos[axis] -= step
                if new_pos[axis] < stop:
                    break
            # Move object towards center
            p.resetBasePositionAndOrientation(obj_id, new_pos, orn)
            p.stepSimulation()
            contact_a = p.getContactPoints(obj_id)
            # If object collides with any other object, stop
            contact_ids = set(item[2]
                              for item in contact_a if item[2] in self.obj_ids)
            if len(contact_ids) != 0:
                collison = True
        # Move one step back
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        new_pos = list(pos)
        if operator == '+':
            new_pos[axis] -= step
        else:
            new_pos[axis] += step
        p.resetBasePositionAndOrientation(obj_id, new_pos, orn)

    def createPacked(self, obj_info):
        init_x, init_y, init_z = self.obj_init_pos[0], self.obj_init_pos[1], self.Z_TABLE_TOP
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[0]
        center_obj, _, _ = self.loadObj(
            path, [init_x, init_y, init_z], yaw, mod_orn, mod_stiffness)

        margin = 0.3
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[1]
        left_obj_id, _, _ = self.loadObj(
            path, [init_x-margin, init_y, init_z], yaw, mod_orn, mod_stiffness)
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[2]
        top_obj_id, _, _ = self.loadObj(
            path, [init_x, init_y+margin, init_z], yaw, mod_orn, mod_stiffness)
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[3]
        right_obj_id, _, _ = self.loadObj(
            path, [init_x+margin, init_y, init_z], yaw, mod_orn, mod_stiffness)
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[4]
        down_obj_id, _, _ = self.loadObj(
            path, [init_x, init_y-margin, init_z], yaw, mod_orn, mod_stiffness)

        self.waittingForAllToBeStable()
        step = 0.01
        self.moveObjectAlongAxis(left_obj_id, 0, '+', step, init_x)
        self.moveObjectAlongAxis(top_obj_id, 1, '-', step, init_y)
        self.moveObjectAlongAxis(right_obj_id, 0, '-', step, init_x)
        self.moveObjectAlongAxis(down_obj_id, 1, '+', step, init_y)
        self.updateObjectStates()

    def shuffleObjects(self,objIDs):
        obj_init_pos = [0.05, -0.52]
        for o in objIDs:
            r_x    = random.uniform(obj_init_pos[0] - 0.1, obj_init_pos[0] + 0.1)
            r_y    = random.uniform(obj_init_pos[1] - 0.1, obj_init_pos[1] + 0.1)
            roll   = random.uniform(0, np.pi)
            orn    = p.getQuaternionFromEuler([roll, 0, 0])
            pos    = [r_x, r_y, 0.85]
            p.resetBasePositionAndOrientation(o,pos,orn)
        

    def creatPileofTube(self,n):
        obj_init_pos = [0.05, -0.52]
        self.tubeObj = []

        for i in range(n):
            r_x    = random.uniform(obj_init_pos[0] - 0.1, obj_init_pos[0] + 0.1)
            r_y    = random.uniform(obj_init_pos[1] - 0.1, obj_init_pos[1] + 0.1)
            roll   = random.uniform(0, np.pi)
            orn    = p.getQuaternionFromEuler([roll, 0, 0])
            pos    = [r_x, r_y, 1]
            obj_id = p.loadURDF("objects/ycb_objects/YcbTomatoSoupCan/model.urdf", pos, orn)
            
            self.dummySimulationSteps(5)
        
            # p.changeDynamics(obj_id, 
            #                 -1,
            #                 spinningFriction = 0.4,
            #                 rollingFriction  = 0.4,
            #                 linearDamping    = 0.0)
            self.tubeObj.append(obj_id)

        self.obj_ids = self.tubeObj    
        self.dummySimulationSteps(100)
        
    def updateEyeInHandCamerPos(self):
        camPosX = p.multiplyTransforms(self.ee_pp,p.getLinkState(self.robot_id,self.eef_id)[1],[0.15,0,0.],p.getQuaternionFromEuler([0,0,0]))
        camPosY = p.multiplyTransforms(self.ee_pp,p.getLinkState(self.robot_id,self.eef_id)[1],[0.,0.15,0.],p.getQuaternionFromEuler([0,0,0]))
        camPosZ = p.multiplyTransforms(self.ee_pp,p.getLinkState(self.robot_id,self.eef_id)[1],[0,0,0.15],p.getQuaternionFromEuler([0,0,0]))
        self.cameraPos = p.multiplyTransforms(self.ee_pp,p.getLinkState(self.robot_id,self.eef_id)[1],[0,0,0.1],p.getQuaternionFromEuler([0,0,0]))

        camLinex = p.addUserDebugLine(self.ee_pp, camPosX[0],[1,0,0], lineWidth=10)
        camLiney = p.addUserDebugLine(self.ee_pp, camPosY[0],[0,1,0], lineWidth=10)
        camLinez = p.addUserDebugLine(self.ee_pp, camPosZ[0],[0,0,1], lineWidth=10)
        time.sleep(0.05)
        p.removeUserDebugItem(camLinex)
        p.removeUserDebugItem(camLiney)
        p.removeUserDebugItem(camLinez)
    
    def getEEState(self):
        return p.getLinkState(self.robot_id,self.eef_id)
        

    def moveEE(self, gPos,gOrn, max_step=1, check_collision_config=None, custom_velocity=None,
                try_close_gripper=False, verbose=False):
        x, y, z = gPos
        orn = gOrn
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        
        for _ in range(max_step):
            # apply IK
            eeState = self.getEEState()
            self.ee_pp   = eeState[0]#p.getLinkState(self.robot_id,self.eef_id)[0]
            self.ee_orn  = p.getEulerFromQuaternion(eeState[1])
            xc,yc,zc = 0.8*np.array(self.ee_pp)+0.2*np.array([x,y,z])
            

            # self.updateEyeInHandCamerPos()

            if self.robotType == "UR5":
                joint_poses = p.calculateInverseKinematics(self.robot_id, self.eef_id, [xc, yc, zc], orn,
                                                        maxNumIterations=100
                                                        )
                # Filter out the gripper
                for i, name in enumerate(self.controlJoints[:-1]):
                    joint = self.joints[name]
                    pose = joint_poses[i]
                    # control robot end-effector
                    p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                            targetPosition=pose, force=joint.maxForce,
                                            maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            else:
                pandaNumDofs = 7

                ll = [-7]*pandaNumDofs
                #upper limits for null space (todo: set them to proper range)
                ul = [7]*pandaNumDofs
                #joint ranges for null space (todo: set them to proper range)
                jr = [7]*pandaNumDofs
                #restposes for null space
                jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
                rp = jointPositions
                joint_poses = p.calculateInverseKinematics(self.robot_id,self.eef_id, [xc, yc, zc], orn, ll, ul,
                    jr, rp, maxNumIterations=20)
                for i in range(pandaNumDofs):
                     p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, joint_poses[i],force=5 * 240.)
    
           
            self.stepSimulation()     
            
            # if try_close_gripper and still_open_flag_ and not self.GripperContact():
            #     still_open_flag_ = self.close_gripper(check_contact=True)
            
            # Check if contact with objects
            if check_collision_config and self.GripperContact(**check_collision_config):
                if self.debug:
                    print('Collision detected!', self.checkGraspedID())
                return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]
            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(
                self.robot_id, self.eef_id)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(
                real_xyzw)
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                if verbose:
                    print('Reach target with', _, 'steps')
                return True, (real_xyz, real_xyzw)

        # raise FailToReachTargetError
        if self.debug:
            print('Failed to reach the target')
        return False, p.getLinkState(self.robot_id, self.eef_id)[0:2]
