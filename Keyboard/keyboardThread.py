import time
from getkey import getkey, keys

class KeyboardThread():
    def __init__(self,freq = 50, lock = None) -> None:
        self.freq = freq
        self.lock = lock
        self.key  = None
        self.ux = 0
        self.uy = 0
        self.uz = 0
        
        self.stepX = 0.005
        self.stepY = 0.005
        self.stepZ = 0.005
        
        self.maxX = 0.2
        self.minX = -0.2
        self.maxY = 0.2
        self.minY = -0.2
        self.maxZ = 0.5
        self.minZ = -0.5
        
        
        
        
    def readkey(self):
        
        while (True):
            
            k = getkey(blocking=True)        
            if k != '':    
                self.key = k
            # time.sleep(1.0/self.freq)

    def updateKeyInfo(self):
        
        key = self.key
        
        if key == keys.UP:
            self.uy = self.uy-self.stepY if self.uy-self.stepY > self.minY else self.minY
        elif  key == keys.DOWN:
            self.uy = self.uy+self.stepY if self.uy+self.stepY < self.maxY else self.maxY
        elif key == keys.LEFT:
            self.ux = self.ux-self.stepX if self.ux-self.stepX > self.minX else self.minX
        elif  key == keys.RIGHT:
            self.ux = self.ux+self.stepX if self.ux+self.stepX < self.maxX else self.maxX    
        elif  key == 'q' or key == 'Q':
            self.uz = self.uz+self.stepZ if self.uz+self.stepZ < self.maxZ else self.maxZ 
        elif  key == 'a' or key == 'A':
            self.uz = self.uz-self.stepZ if self.uz-self.stepZ > self.minZ else self.minZ
               
        
        elif  key == 'z' or key == 'Z':
             self.uy = 0
        elif  key == 'x' or key == 'X':
             self.ux = 0     
        elif  key == 'c' or key == 'C':
             self.uz = 0     
             
        elif key == ' ':
            self.ux = 0.8*self.ux
            self.uy = 0.8*self.uy
            self.uz = 0.8*self.uz
            
        elif key == 'v' or key == 'v':
            self.ux = 0
            self.uy = 0
            self.uy = 0
            
        
        self.key = ''
        return self.ux, self.uy, self.uz, key

    def waitForKey(self,listofKey):
        key = self.key
        while  key not in listofKey:
            key = self.key
            time.sleep(0.05)
        self.key = ''
        return key

        


           

