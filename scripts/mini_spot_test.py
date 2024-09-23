import pybullet as p
import pybullet_data as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scripts.CPG import CPG

from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment


class A1Env():
    def __init__(self, gui = True) -> None:            
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        self.bullet = p
        self._samplingTime = 0.005
        p.setTimeStep(self._samplingTime)

        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0,0,-9.81)

        self.FloorId = p.loadURDF("plane.urdf",[0,0,-0.])        
        self.robotID = p.loadURDF('a1/a1.urdf', [0,0,0.44])
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=180, cameraPitch=-35,
        #                             cameraTargetPosition=[0., 0, 0.1])
    
        #Lock joints in place
        numJoints = p.getNumJoints(self.robotID)
        for j in range(numJoints):
            p.setJointMotorControl2( bodyIndex = self.robotID, jointIndex = j, controlMode = p.POSITION_CONTROL, targetPosition = 0 )
            info = p.getJointInfo(self.robotID, j)
            print (info)

        self.tsim = 0
        self.JointPositions = np.zeros(12)    
        
        self.reset(zleg = -0.25)

    def add_harmony_box(self, box_centre,ori_offset = [0.0, 0.0, 0.]):
        id1 = p.loadURDF("environment_Yumi/urdf/objects/box.urdf",
                         box_centre,
                         p.getQuaternionFromEuler(ori_offset),
                         useFixedBase=True)
        
    def _EndEffectorIK(self, leg_id, position, position_in_world_frame):
        """Calculate the joint positions from the end effector position."""
        assert len(self._foot_link_ids) == self.num_legs
        toe_id = self._foot_link_ids[leg_id]
        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = [
            i for i in range(leg_id * motors_per_leg, leg_id * motors_per_leg +
                            motors_per_leg)
        ]
        joint_angles = self.joint_angles_from_link_position(
            robot=self,
            link_position=position,
            link_id=toe_id,
            joint_ids=joint_position_idxs,
            position_in_world_frame=position_in_world_frame)
        # Joint offset is necessary for A1.
        joint_angles = np.multiply(
            np.asarray(joint_angles) -
            np.asarray(self._motor_offset)[joint_position_idxs],
            self._motor_direction[joint_position_idxs])
        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()
                            
    def get_ee_state(self):
        pose = p.getLinkState(self.robotID,0)[0:2]        
        return pose[0] , pose[1]
        
    def reset(self,zleg):
        stabilization_steps = 1000
        p.setTimeStep(self._samplingTime)
        for _ in range(stabilization_steps): 
            p.stepSimulation()
            pFL = np.array((0.0,0.0,zleg))
            pFR = np.array((0.0,-0.0,zleg))
            pBL = np.array((0.0,0.0,zleg))
            pBR = np.array((0.0,-0.0,zleg))
            FL = self.IK(pFL)
            FR = self.IK(pFR)
            BL = self.IK(pBL)
            BR = self.IK(pBR)
            self.JointPositions[0:3] = FL
            self.JointPositions[3:6] = FR
            self.JointPositions[6:9] = BL
            self.JointPositions[9:12]= BR
            self.applyMotorCommand()

    def IK(self,targetPosition):

        Lu = 0.208
        Ld = 0.203
        epsilon = 0.000001

        Lx2 =  (0*targetPosition[0]**2) + (targetPosition[2]**2)
        Ly2 =  (targetPosition[1]**2) + (targetPosition[2]**2)

        Lx = np.sqrt(Lx2)
        Ly = np.sqrt(Ly2)

        Lu2 = Lu ** 2
        Ld2 = Ld ** 2

        alpha = (Lu2+Ld2-Lx2)/((2*Lu*Ld)+epsilon)
        if alpha>1: 
            alpha =1 
        elif (alpha<-1):
            alpha=-1

        thetaKnee  = np.arccos(alpha) - np.pi
        
        beta = (Lu2+Lx2-Ld2)/(2*Lu*Lx+epsilon)
        if (beta>1):
            beta = 1
        elif (beta<-1):
            beta = -1


        if (targetPosition[0]>=0):
            thetaHipx  = np.arccos(beta) + np.arctan(np.abs(targetPosition[0]) / (Lx+epsilon))
        else:
            thetaHipx  = np.arccos(beta) - np.arctan(np.abs(targetPosition[0]) / (Lx+epsilon))
            

        thetaHipy  = np.arctan(targetPosition[1] / (Ly+epsilon))
        JointPoses = [thetaHipy,thetaHipx,thetaKnee]
        
        return JointPoses 

    def applyMotorCommand(self):
        # FR [1,3,4]
        # FL [6,8,9]
        # RR [11,13,14]
        # RL [16,18,19]
        motor_id = [1,3,4,6,8,9,11,13,14,16,18,19]
        for i in range(len(self.JointPositions)):
            p.setJointMotorControl2(bodyIndex=self.robotID, jointIndex=motor_id[i], controlMode=p.POSITION_CONTROL,
                                                targetPosition=self.JointPositions[i], force=200)


    def step(self,cpg):

        if cpg.gtime < 5:
            cpg.NewStepX_raw = -0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0
        elif cpg.gtime < 20:
            cpg.NewStepX_raw = -0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        elif cpg.gtime < 25:
            cpg.NewStepX_raw = -0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        elif cpg.gtime < 30:
            cpg.NewStepX_raw = -0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        


    def add_a_cube(self,pos,size=[0.1,0.1,0.1],mass = 0.1, color = [1,1,0,1], textureUniqueId = None):

        # cubesID = []
        box     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])
        vis     = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=color)
        obj_id  = p.createMultiBody(mass, box, vis, pos, [0,0,0,1])
        p.changeDynamics(obj_id, 
                        -1,
                        spinningFriction=800,
                        rollingFriction=0.0,
                        linearDamping=50.0)
        
        if textureUniqueId is not None:
            p.changeVisualShape(obj_id, -1, textureUniqueId=textureUniqueId)

        # cubesID.append(obj_id)
        
        p.stepSimulation()
        return obj_id 



        # cpg.gtime = self.tsim
        cpg.apply_walk_command()
        cpg.updateOmniJoints_CPG()
        
        StepTheta = cpg.NewStepTheta_raw            

        pFL = np.array([-0.0 + cpg.LfootPosition[0], 0.00 + cpg.LfootPosition[1], 0.00 + cpg.LfootPosition[2]])
        pFR = np.array([-0.0 + cpg.RfootPosition[0], 0.00 + cpg.RfootPosition[1], 0.00 + cpg.RfootPosition[2]])

        pBL = np.array([-0.0 + cpg.RfootPosition[0],-0.00 + cpg.RfootPosition[1], 0.00 +cpg.RfootPosition[2]])
        pBR = np.array([-0.0 + cpg.LfootPosition[0],-0.00 + cpg.LfootPosition[1], 0.00 + cpg.LfootPosition[2]])


        pFL[0] = pFL[0]*np.cos(StepTheta)-pFL[1]*np.sin(StepTheta)
        pFL[1] = pFL[0]*np.sin(StepTheta)+pFL[1]*np.cos(StepTheta)

        pFR[0] = pFR[0]*np.cos(StepTheta)-pFR[1]*np.sin(StepTheta)
        pFR[1] = pFR[0]*np.sin(StepTheta)+pFR[1]*np.cos(StepTheta)

        pBL[0] = pBL[0]*np.cos(StepTheta)-pBL[1]*np.sin(StepTheta)
        pBL[1] = pBL[0]*np.sin(StepTheta)+pBL[1]*np.cos(StepTheta)

        pBR[0] = pBR[0]*np.cos(StepTheta)-pBR[1]*np.sin(StepTheta)
        pBR[1] = pBR[0]*np.sin(StepTheta)+pBR[1]*np.cos(StepTheta)
        
        
        FR = self.IK(pFL)
        FL = self.IK(pFR)
        BR = self.IK(pBL)
        BL = self.IK(pBR)
        self.JointPositions[0:3] = FR
        self.JointPositions[3:6] = FL
        self.JointPositions[6:9] = BR
        self.JointPositions[9:12]= BL



        self.applyMotorCommand()
        
        p.stepSimulation()
        self.tsim += self._samplingTime
        time.sleep(self._samplingTime)

        print ("Finished")


        


class MiniSpotEnv():
    def __init__(self) -> None:            
        self.physicsClient = p.connect(p.GUI)
        self._pybullet = p
        self._samplingTime = 0.005
        p.setTimeStep(self._samplingTime)

        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0,0,-9.81)

        self.FloorId = p.loadURDF("plane.urdf",[0,0,-0.])        
        robot_a1= p.loadURDF('a1/a1.urdf', [0,1,0.35])
        self.robotID = p.loadURDF("environment/urdf_models/spotmicro_proprio_v5/urdf/spotmicro_proprio_v5.urdf",[0,0,0.2])

        boundaries = p.getAABB(self.robotID,1)


        self.colorPalettes = {
            "lightOrange": [1.0, 0.82, 0.12, 1.0],
            "darkOrange": [1.0, 0.6, 0.0, 1.0],
            "darkGrey": [0.43, 0.43, 0.43, 1.0],
            "lightGrey": [0.65, 0.65, 0.65, 1.0],
        }


        p.resetVisualShapeData(
                    self.robotID, 0, 
                    rgbaColor=self.colorPalettes["darkOrange"])

        p.resetVisualShapeData(
                    self.robotID, 3, 
                    rgbaColor=self.colorPalettes["darkOrange"])



        #Lock joints in place
        numJoints = p.getNumJoints(self.robotID)
        for j in range(numJoints):
            p.setJointMotorControl2( bodyIndex = self.robotID, jointIndex = j, controlMode = p.POSITION_CONTROL, targetPosition = 0 )
            aabb = p.getAABB(self.robotID, j)
            aabbMin = np.array(aabb[0])
            aabbMax = np.array(aabb[1])
            print(aabbMax - aabbMin)
            
            # print(aabbMin)
            # print(aabbMax)
        
            info = p.getJointInfo(self.robotID, j)
            print (info)

        self.tsim = 0
        self.JointPositions = np.zeros(12)    
        
        self.reset(-0.23)

                            
    def get_ee_state(self):
        pose = p.getLinkState(self.robotID,0)[0:2]        
        return pose[0] , pose[1]
        
    def reset(self,zleg):
        stabilization_steps = 1000
        p.setTimeStep(self._samplingTime)
        for _ in range(stabilization_steps): 
            p.stepSimulation()
            pFL = np.array((0.0,0.0,zleg))
            pFR = np.array((0.0,-0.0,zleg))
            pBL = np.array((0.0,0.0,zleg))
            pBR = np.array((0.0,-0.0,zleg))
            FL = self.IK(pFL)
            FR = self.IK(pFR)
            BL = self.IK(pBL)
            BR = self.IK(pBR)
            self.JointPositions[0:3] = FL
            self.JointPositions[3:6] = FR
            self.JointPositions[6:9] = BL
            self.JointPositions[9:12]= BR
            self.applyMotorCommand()

    def IK(self,targetPosition):

        Lu = 0.118
        Ld = 0.113
        epsilon = 0.000001

        Lx2 =  (0*targetPosition[0]**2) + (targetPosition[2]**2)
        Ly2 =  (targetPosition[1]**2) + (targetPosition[2]**2)

        Lx = np.sqrt(Lx2)
        Ly = np.sqrt(Ly2)

        Lu2 = Lu ** 2
        Ld2 = Ld ** 2

        alpha = (Lu2+Ld2-Lx2)/((2*Lu*Ld)+epsilon)
        if alpha>1: 
            alpha =1 
        elif (alpha<-1):
            alpha=-1

        thetaKnee  = np.arccos(alpha) - np.pi
        
        beta = (Lu2+Lx2-Ld2)/(2*Lu*Lx+epsilon)
        if (beta>1):
            beta = 1
        elif (beta<-1):
            beta = -1


        if (targetPosition[0]>=0):
            thetaHipx  = np.arccos(beta) + np.arctan(np.abs(targetPosition[0]) / (Lx+epsilon))
        else:
            thetaHipx  = np.arccos(beta) - np.arctan(np.abs(targetPosition[0]) / (Lx+epsilon))
            

        thetaHipy  = np.arctan(targetPosition[1] / (Ly+epsilon))
        JointPoses = [thetaHipy,thetaHipx,thetaKnee]
        
        return JointPoses 

    def applyMotorCommand(self):
        for i in range(len(self.JointPositions)):
            p.setJointMotorControl2(bodyIndex=self.robotID, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                                targetPosition=self.JointPositions[i], force=20)


    
    def step(self,cpg):

        if cpg.gtime < 5:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0
        elif cpg.gtime < 20:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        elif cpg.gtime < 25:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        elif cpg.gtime < 30:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        





        cpg.gtime = self.tsim
        cpg.apply_walk_command()
        cpg.updateOmniJoints_CPG()
        
        StepTheta = cpg.NewStepTheta_raw            

        pFL = np.array([-0.0 + cpg.LfootPosition[0], 0.00 + cpg.LfootPosition[1], 0.00 + cpg.LfootPosition[2]])
        pFR = np.array([-0.0 + cpg.RfootPosition[0], 0.00 + cpg.RfootPosition[1], 0.00 + cpg.RfootPosition[2]])

        pBL = np.array([-0.0 + cpg.RfootPosition[0],-0.00 + cpg.RfootPosition[1], 0.00 +cpg.RfootPosition[2]])
        pBR = np.array([-0.0 + cpg.LfootPosition[0],-0.00 + cpg.LfootPosition[1], 0.00 + cpg.LfootPosition[2]])


        pFL[0] = pFL[0]*np.cos(StepTheta)-pFL[1]*np.sin(StepTheta)
        pFL[1] = pFL[0]*np.sin(StepTheta)+pFL[1]*np.cos(StepTheta)

        pFR[0] = pFR[0]*np.cos(StepTheta)-pFR[1]*np.sin(StepTheta)
        pFR[1] = pFR[0]*np.sin(StepTheta)+pFR[1]*np.cos(StepTheta)

        pBL[0] = pBL[0]*np.cos(StepTheta)-pBL[1]*np.sin(StepTheta)
        pBL[1] = pBL[0]*np.sin(StepTheta)+pBL[1]*np.cos(StepTheta)

        pBR[0] = pBR[0]*np.cos(StepTheta)-pBR[1]*np.sin(StepTheta)
        pBR[1] = pBR[0]*np.sin(StepTheta)+pBR[1]*np.cos(StepTheta)
        
        
        FL = self.IK(pFL)
        FR = self.IK(pFR)
        BL = self.IK(pBL)
        BR = self.IK(pBR)
        self.JointPositions[0:3] = FR
        self.JointPositions[3:6] = FL
        self.JointPositions[6:9] = BR
        self.JointPositions[9:12]= BL



        self.applyMotorCommand()
        
        p.stepSimulation()
        self.tsim += self._samplingTime
        time.sleep(self._samplingTime)

        print ("Finished")


        


if __name__ == "__main__":
    env = A1Env()
    cpg = CPG(Zleg = -0.3)
    cpg.NewStepX_raw = 0.0
    cpg.NewStepY_raw = 0.0
    cpg.NewStepTheta_raw = 0
    env.add_harmony_box([0.4,0.14,0])

    
    soft_robot_1 = SoftRobotBasicEnvironment( bullet = env.bullet,number_of_segment=4)
    base_link_id = None

    env.add_a_cube([0.7,0.1,0.3],[0.3,0.4,0.02],mass=0.1,color=[0.7,0.3,0.4,1])

    
    t = 0
    dt = 0.01
    cam_pos = np.array([0,0,0])
    while True:    
        t += dt
        env.step(cpg)
    
        
        # sf1_seg1_cable_1   = .003*np.sin(0.5*np.pi*t)
        # sf1_seg1_cable_2   = 0.01+.005*np.sin(0.5*np.pi*t)
        # sf1_seg2_cable_1   = 0.005 + .00*np.sin(0.5*np.pi*t+1)
        # sf1_seg2_cable_2   = 0.005+.003*np.sin(0.5*np.pi*t+1)
        # sf1_seg3_cable_0   = .02*np.sin(0.5*np.pi*t)
        # sf1_seg3_cable_1   = .01*np.sin(0.5*np.pi*t+2)
        # sf1_seg3_cable_2   = .02*np.sin(0.5*np.pi*t+2)
        # sf1_gripper_pos    = np.abs(np.sin(np.pi*t))
                
        sf1_seg1_cable_1   = 0.0
        sf1_seg1_cable_2   = 0.01
        sf1_seg2_cable_1   = 0.01 
        sf1_seg2_cable_2   = 0.0
        sf1_seg3_cable_0   = 0.02
        sf1_seg3_cable_1   = -0.005
        sf1_seg3_cable_2   = -0.00
        sf1_gripper_pos    = 0.00
        
        p0,o0 = env.get_ee_state()
        p0,o0 = env.bullet.multiplyTransforms(p0, o0, [0.23, 0.0,0.1], [0,0,0,1])
        angle = np.pi
        rotation_quaternion = env.bullet.getQuaternionFromEuler([angle, 0, angle/2])
        
        new_pos, new_ori = env.bullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
        base_orin = env.bullet.getEulerFromQuaternion(new_ori)
        if base_link_id is None:
            base_link_shape = env.bullet.createVisualShape(env.bullet.GEOM_BOX, halfExtents=[0.025, 0.025, 0.03], rgbaColor=[0.6, 0.6, 0.6, 1])
            base_link_pos, base_link_ori = env.bullet.multiplyTransforms(new_pos, new_ori, [0,-0.02,0], [0,0,0,1])
            base_link_id    = env.bullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_link_shape,
                                                        baseVisualShapeIndex=base_link_shape,
                                                        basePosition= base_link_pos , baseOrientation=base_link_ori)
        else:
            base_link_pos, base_link_ori = env.bullet.multiplyTransforms(new_pos, new_ori, [0,-0.02,0.0], [0,0,0,1])
            env.bullet.resetBasePositionAndOrientation(base_link_id, base_link_pos , base_link_ori)
        
        # cam_pos = 0.8*cam_pos + 0.2*np.array([p0[0],0,0.2])
        
        # env._pybullet.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=cam_pos)

        soft_robot_1.move_robot_ori(action=np.array([sf1_seg3_cable_0, sf1_seg1_cable_1, sf1_seg1_cable_2, 
                                                    0.0, sf1_seg2_cable_1, sf1_seg2_cable_2,
                                                    sf1_seg3_cable_0, sf1_seg3_cable_1, sf1_seg3_cable_2]),
                                base_pos = new_pos, base_orin = base_orin)
        
