import numpy as np
import math

import matplotlib.pyplot as plt


class CPG():
    def __init__(self,Zleg = -0.18) -> None:

        self.gtime = 0
        self.SamplingTime=0.001

        self.StepX= 0.0
        self.StepY= 0.0
        self.StepTheta = 0.0
        self.SwingStepZ = 0.02
        self.StepTime = 0.3
        self.Zleg = Zleg 
        self.update_time_param()
        self.update_move_param()   
        self.generate_cpg(0)
        self.twalk0 = 0


        
        self.NewStepX_raw = 0
        self.NewStepY_raw = 0
        self.NewStepTheta_raw  = 0
        self.NewStepTime = np.copy(self.StepTime)
        
        self.NewStepX = 0
        self.NewStepY = 0
        self.NewStepTheta  = 0
        
        
        self.NewCommand = False



    def wsin(self,time, period, period_shift, mag, mag_shift):
            return mag * np.sin(2 * np.pi / period * time - period_shift) + mag_shift

    def update_time_param(self):

        self.DSP_Ratio      = 0.1
        self.PELVIS_OFFSET  = 0.0
        self.ARM_SWING_GAIN = 3.5
        
        
        SSP_Ratio              = 1 - self.DSP_Ratio

        self.X_Swap_PeriodTime = self.StepTime
        self.X_Move_PeriodTime = 2 * self.StepTime * SSP_Ratio

        self.Y_Swap_PeriodTime = 2 * self.StepTime
        self.Y_Move_PeriodTime = 2 * self.StepTime * SSP_Ratio
        
        self.Z_Swap_PeriodTime = self.StepTime
        self.Z_Move_PeriodTime = self.StepTime * SSP_Ratio
        
        self.A_Move_PeriodTime = 2 * self.StepTime * SSP_Ratio

        self.SSP_Time          = 2 * self.StepTime * SSP_Ratio
        self.SSP_Time_Start_L  = (1 - SSP_Ratio) * self.StepTime / 2
        self.SSP_Time_End_L    = (1 + SSP_Ratio) * self.StepTime / 2
        self.SSP_Time_Start_R  = (3 - SSP_Ratio) * self.StepTime / 2
        self.SSP_Time_End_R    = (3 + SSP_Ratio) * self.StepTime / 2

        self.Phase_Time1       = (self.SSP_Time_End_L + self.SSP_Time_Start_L) / 2
        self.Phase_Time2       = (self.SSP_Time_Start_R + self.SSP_Time_End_L) / 2
        self.Phase_Time3       = (self.SSP_Time_End_R + self.SSP_Time_Start_R) / 2

        self.Pelvis_Offset     = self.PELVIS_OFFSET
        self.Pelvis_Swing      = self.Pelvis_Offset * 0.35
        self.Arm_Swing_Gain    = self.ARM_SWING_GAIN    


    def update_move_param(self):
        # Forward/Back
        self.X_Move_Amplitude = self.StepX / 2
        self.X_Swap_Amplitude = self.StepX / 4
        
        # Right/Left
        self.Y_Move_Amplitude = self.StepY / 2
        if ( self.Y_Move_Amplitude > 0):
            self.Y_Move_Amplitude_Shift = self.Y_Move_Amplitude
        else:
            self.Y_Move_Amplitude_Shift = -self.Y_Move_Amplitude
    
        self.Y_SWAP_AMPLITUDE = 0.005
        self.Y_Swap_Amplitude = self.Y_SWAP_AMPLITUDE + (self.Y_Move_Amplitude_Shift * 0.005)

        
        # self.Theta_Move_Amplitude = math.radians(self.StepTheta) / 2

        self.Z_Move_Amplitude = self.SwingStepZ / 1
        self.Z_SWAP_AMPLITUDE = 0.0005
        self.Z_Move_Amplitude_Shift = self.Z_Move_Amplitude / 2
        self.Z_Swap_Amplitude = self.Z_SWAP_AMPLITUDE
        self.Z_Swap_Amplitude_Shift = self.Z_Swap_Amplitude

        # Theta movement
        self.A_MOVEMENT_ON = False
        self.A_MOVE_AMPLITUDE = self.StepTheta / 2
        if(self.A_MOVEMENT_ON == False): 
            self.A_Move_Amplitude = self.A_MOVE_AMPLITUDE / 2
            if(self.A_Move_Amplitude > 0):
                self.A_Move_Amplitude_Shift = self.A_Move_Amplitude
            else:
                self.A_Move_Amplitude_Shift = -self.A_Move_Amplitude
        else:    
            self.A_Move_Amplitude = -self.A_MOVE_AMPLITUDE / 2
            if( self.A_Move_Amplitude > 0 ):
                self.A_Move_Amplitude_Shift = -self.A_Move_Amplitude
            else:
                self.A_Move_Amplitude_Shift = self.A_Move_Amplitude
    

    def apply_walk_command(self):
    
        self.NewStepX +=     max( min( self.NewStepX_raw - self.NewStepX, 0.05) , -0.05)
        self.NewStepY +=     max( min( self.NewStepY_raw - self.NewStepY, 0.025) , -0.025)
        self.NewStepTheta += max( min( self.NewStepTheta_raw - self.NewStepTheta, np.pi/10) , -np.pi/10)

        self.NewCommand = 1

        #print("Apply:", self.NewStepX, self.NewStepY, self.NewStepTheta)
        
    def generate_cpg(self,Time):

        #Time = (int(time*1000))  % int(2 * self.StepTime * 1000)
        #Time = t % self.StepTime 
        TIME_UNIT = self.SamplingTime / 2

        self.X_Swap_Phase_Shift     = np.pi
        self.X_Swap_Amplitude_Shift = 0
        self.X_Move_Phase_Shift     = np.pi / 2
        self.X_Move_Amplitude_Shift = 0
        
        self.Y_Swap_Phase_Shift     = 0
        self.Y_Swap_Amplitude_Shift = 0
        self.Y_Move_Phase_Shift     = np.pi / 2
            
        self.Z_Swap_Phase_Shift = np.pi * 3 / 2
        self.Z_Move_Phase_Shift = np.pi / 2
            
        self.A_Move_Phase_Shift = np.pi / 2

        if (Time <= TIME_UNIT):
            self.update_time_param()
        
        elif(Time >= (self.Phase_Time1 - TIME_UNIT) and Time < (self.Phase_Time1 + TIME_UNIT)):    
            self.update_move_param()
        
        elif(Time >= (self.Phase_Time2 - TIME_UNIT) and Time < (self.Phase_Time2 + TIME_UNIT)):
            self.update_time_param()
            # Time = self.Phase_Time2

        elif(Time >= (self.Phase_Time3 - TIME_UNIT) and Time < (self.Phase_Time3 + TIME_UNIT)):    
            self.update_move_param()
        
        x_swap = self.wsin(Time, self.X_Swap_PeriodTime, self.X_Swap_Phase_Shift, self.X_Swap_Amplitude, self.X_Swap_Amplitude_Shift)
        y_swap = self.wsin(Time, self.Y_Swap_PeriodTime, self.Y_Swap_Phase_Shift, self.Y_Swap_Amplitude, self.Y_Swap_Amplitude_Shift)
        z_swap = self.wsin(Time, self.Z_Swap_PeriodTime, self.Z_Swap_Phase_Shift, self.Z_Swap_Amplitude, self.Z_Swap_Amplitude_Shift)
        c_swap = z_swap

        if (Time <= self.SSP_Time_Start_L): 
            x_move_l = self.wsin(self.SSP_Time_Start_L, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_L, self.X_Move_Amplitude, self.X_Move_Amplitude_Shift)
            y_move_l = self.wsin(self.SSP_Time_Start_L, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_L, self.Y_Move_Amplitude, self.Y_Move_Amplitude_Shift)
            z_move_l = self.wsin(self.SSP_Time_Start_L, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_L, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_l = self.wsin(self.SSP_Time_Start_L, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_L, self.A_Move_Amplitude, self.A_Move_Amplitude_Shift)

            x_move_r = self.wsin(self.SSP_Time_Start_L, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_L, -self.X_Move_Amplitude, -self.X_Move_Amplitude_Shift)
            y_move_r = self.wsin(self.SSP_Time_Start_L, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_L, -self.Y_Move_Amplitude, -self.Y_Move_Amplitude_Shift)
            z_move_r = self.wsin(self.SSP_Time_Start_R, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_R,  self.Z_Move_Amplitude,   self.Z_Move_Amplitude_Shift)
            c_move_r = self.wsin(self.SSP_Time_Start_L, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_L, -self.A_Move_Amplitude, -self.A_Move_Amplitude_Shift)

            pelvis_offset_l = 0
            pelvis_offset_r = 0            
        
        elif (Time <= self.SSP_Time_End_L):    
            x_move_l = self.wsin(Time, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_L, self.X_Move_Amplitude, self.X_Move_Amplitude_Shift)
            y_move_l = self.wsin(Time, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_L, self.Y_Move_Amplitude, self.Y_Move_Amplitude_Shift)
            z_move_l = self.wsin(Time, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_L, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_l = self.wsin(Time, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_L, self.A_Move_Amplitude, self.A_Move_Amplitude_Shift)

            x_move_r = self.wsin(Time, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_L, -self.X_Move_Amplitude, -self.X_Move_Amplitude_Shift)
            y_move_r = self.wsin(Time, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_L, -self.Y_Move_Amplitude, -self.Y_Move_Amplitude_Shift)
            z_move_r = self.wsin(self.SSP_Time_Start_R, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_R, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_r = self.wsin(Time, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_L, -self.A_Move_Amplitude, -self.A_Move_Amplitude_Shift)
            
            pelvis_offset_l = self.wsin(Time, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_L, self.Pelvis_Swing / 2, self.Pelvis_Swing / 2)
            pelvis_offset_r = self.wsin(Time, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_L, -self.Pelvis_Offset / 2, -self.Pelvis_Offset / 2)
        
        elif (Time <= self.SSP_Time_Start_R):    
            x_move_l = self.wsin(self.SSP_Time_End_L, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_L, self.X_Move_Amplitude, self.X_Move_Amplitude_Shift)
            y_move_l = self.wsin(self.SSP_Time_End_L, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_L, self.Y_Move_Amplitude, self.Y_Move_Amplitude_Shift)
            z_move_l = self.wsin(self.SSP_Time_End_L, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_L, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_l = self.wsin(self.SSP_Time_End_L, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_L, self.A_Move_Amplitude, self.A_Move_Amplitude_Shift)
            
            x_move_r = self.wsin(self.SSP_Time_End_L, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_L, -self.X_Move_Amplitude, -self.X_Move_Amplitude_Shift)
            y_move_r = self.wsin(self.SSP_Time_End_L, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_L, -self.Y_Move_Amplitude, -self.Y_Move_Amplitude_Shift)
            z_move_r = self.wsin(self.SSP_Time_Start_R, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_R, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_r = self.wsin(self.SSP_Time_End_L, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_L, -self.A_Move_Amplitude, -self.A_Move_Amplitude_Shift)
            
            pelvis_offset_l = 0
            pelvis_offset_r = 0    
        
        elif( Time <= self.SSP_Time_End_R):    

            x_move_l = self.wsin(Time, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, self.X_Move_Amplitude, self.X_Move_Amplitude_Shift)
            y_move_l = self.wsin(Time, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, self.Y_Move_Amplitude, self.Y_Move_Amplitude_Shift)
            z_move_l = self.wsin(self.SSP_Time_End_L, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_L, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_l = self.wsin(Time, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, self.A_Move_Amplitude, self.A_Move_Amplitude_Shift)
            
            x_move_r = self.wsin(Time, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, -self.X_Move_Amplitude, -self.X_Move_Amplitude_Shift)
            y_move_r = self.wsin(Time, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, -self.Y_Move_Amplitude, -self.Y_Move_Amplitude_Shift)
            z_move_r = self.wsin(Time, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_R, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_r = self.wsin(Time, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, -self.A_Move_Amplitude, -self.A_Move_Amplitude_Shift)
            
            pelvis_offset_l = self.wsin(Time, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_R, self.Pelvis_Offset / 2, self.Pelvis_Offset / 2)
            pelvis_offset_r = self.wsin(Time, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_R, -self.Pelvis_Swing / 2, -self.Pelvis_Swing / 2)
        
        else:    
        
            x_move_l = self.wsin(self.SSP_Time_End_R, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, self.X_Move_Amplitude, self.X_Move_Amplitude_Shift)
            y_move_l = self.wsin(self.SSP_Time_End_R, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, self.Y_Move_Amplitude, self.Y_Move_Amplitude_Shift)
            z_move_l = self.wsin(self.SSP_Time_End_L, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_L, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_l = self.wsin(self.SSP_Time_End_R, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, self.A_Move_Amplitude, self.A_Move_Amplitude_Shift)
            
            x_move_r = self.wsin(self.SSP_Time_End_R, self.X_Move_PeriodTime, self.X_Move_Phase_Shift + 2 * np.pi / self.X_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, -self.X_Move_Amplitude, -self.X_Move_Amplitude_Shift)
            y_move_r = self.wsin(self.SSP_Time_End_R, self.Y_Move_PeriodTime, self.Y_Move_Phase_Shift + 2 * np.pi / self.Y_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, -self.Y_Move_Amplitude, -self.Y_Move_Amplitude_Shift)
            z_move_r = self.wsin(self.SSP_Time_End_R, self.Z_Move_PeriodTime, self.Z_Move_Phase_Shift + 2 * np.pi / self.Z_Move_PeriodTime * self.SSP_Time_Start_R, self.Z_Move_Amplitude, self.Z_Move_Amplitude_Shift)
            c_move_r = self.wsin(self.SSP_Time_End_R, self.A_Move_PeriodTime, self.A_Move_Phase_Shift + 2 * np.pi / self.A_Move_PeriodTime * self.SSP_Time_Start_R + np.pi, -self.A_Move_Amplitude, -self.A_Move_Amplitude_Shift)
            
            pelvis_offset_l = 0
            pelvis_offset_r = 0
        
        if (self.X_Move_Amplitude == 0):
            arm_r = 0 
            arm_l = 0 
        else:
            arm_r = self.wsin(Time, self.StepTime * 2, np.pi * 1.5, -self.X_Move_Amplitude * self.Arm_Swing_Gain, 0)
            arm_l = self.wsin(Time, self.StepTime * 2, np.pi * 1.5, self.X_Move_Amplitude * self.Arm_Swing_Gain, 0)


        xl  = x_swap + x_move_l
        yl  = y_swap + y_move_l  
        zl  = self.Zleg + z_swap + z_move_l
        tl  = c_move_l + c_swap
            
        xr  = x_swap + x_move_r
        yr  = y_swap + y_move_r
        zr  = self.Zleg  + z_swap + z_move_r
        tr  = c_move_r + c_swap
        
        l_pos = np.array([xl,yl,zl,tl,arm_l])
        r_pos = np.array([xr,yr,zr,tr,arm_r])   

        return l_pos, r_pos


    def updateOmniJoints_CPG(self):

        if (abs(self.gtime - self.twalk0) > (2*self.StepTime) - (self.SamplingTime/2)):
            self.twalk0 = self.gtime
            self.StopForOneStep =0

            if (self.NewCommand): 
                self.StepTheta = self.NewStepTheta
                self.StepX = self.NewStepX*math.cos(self.StepTheta)-self.NewStepY*math.sin(self.StepTheta)
                self.StepY = self.NewStepX*math.sin(self.StepTheta)+self.NewStepY*math.cos(self.StepTheta)
                self.StepTime = self.NewStepTime
                self.NewCommand = 0
                print (f"x:{self.StepX:3.3f}\t y:{self.StepY:3.3f}\t theta:{self.StepTheta:3.3f}")

            self.idx = -1
            self.update_time_param()
            self.update_move_param()
            self.generate_cpg(0)

        else:
            
            twalk = self.gtime - self.twalk0
            
            lpos, rpos = self.generate_cpg(twalk)
            self.RfootPosition = rpos
            self.LfootPosition = lpos
            

if __name__ == "__main__":
    cpg = CPG()
    cpg.StepX = 0.02
    cpg.StepY = 0.01
    
    cpg.update_time_param()
    cpg.update_move_param()   
    cpg.generate_cpg(0)

    for i in range(10000):
        cpg.gtime = i*cpg.SamplingTime
        cpg.updateOmniJoints_CPG()
        # plt.plot(i,cpg.LfootPosition[2],'ro')
        # plt.plot(i,cpg.RfootPosition[2],'bo')
        if i%25 == 0:
            plt.plot(i*cpg.SamplingTime,cpg.LfootPosition[0],'ro')
            plt.plot(i*cpg.SamplingTime,cpg.RfootPosition[0],'bo')

            plt.pause(0.001)





