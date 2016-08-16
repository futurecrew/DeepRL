import random
import numpy as np

class Tester:
    def __init__(self):
        self.totalState = 10
        self.epsilon = 0.1 
        self.stepSize = 0.25
        self.discount = 1.0 - 1.0 / self.totalState
        self.maxIter = 10**6
        self.repeatNo = 5
        
    def initialize(self):
        self.Qval = np.random.normal(0, 0.1, (self.totalState, 2)).astype(np.float32)
        self.leftActions = np.random.random_integers(0, 1, self.totalState)
        
    def getAction(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 1) 
        else:
            return np.argmax(self.getQval(s))
    
    def getQval(self, s, a=None):
        if a == None:
            return self.Qval[s]
        else:
            return self.Qval[s, a]
        
    def isLeftAction(self, s, a):
        if self.leftActions[s] == a:
            return True
        else:
            return False
        
    def doAction(self, s, a):
        """ returns next state and reward """
         
        if self.isLeftAction(s, a):
            return 0, 0
        else:
            if s < self.totalState - 1:
                return s+1, 0
            else:
                return 0, 1
            
    def isComplete(self):
        for s in range(self.totalState):
            policy = np.argmax(self.getQval(s))
            if self.isLeftAction(s, policy) == True:
                return False
        return True
    
    def gogo(self):        
        print 'Training starts'

        trainDone = []
        for repeat in range(self.repeatNo):
            s = 0
            self.initialize()
            
            for i in range(self.maxIter):
                a = self.getAction(s)
                s2, r = self.doAction(s, a)
                td = r + self.discount * np.max(self.getQval(s2)) - self.getQval(s, a)
                self.Qval[s, a] = self.getQval(s, a) + self.stepSize * td
                s = s2
                
                if i % 10 == 0:
                    if self.isComplete():
                        trainDone.append(i)
                        break
        
        print '%s' % trainDone
        print '%s training complete. mean iters = %.0f' % (self.repeatNo, np.mean(trainDone))
        
if __name__ == '__main__':
    Tester().gogo()