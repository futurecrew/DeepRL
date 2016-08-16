import random
import numpy as np

class Tester:
    def __init__(self):
        self.totalState = 5
        self.totalAction = 2
        self.epsilon = 0.1 
        self.stepSize = 0.25
        self.discount = 1.0 - 1.0 / self.totalState
        self.maxIter = 10**6
        self.repeatNo = 10
        
        self.mode = 'FA'
        #self.mode = 'Tablar'
        
    def initialize(self):
        self.Qval = np.random.normal(0, 0.1, (self.totalState, self.totalAction)).astype(np.float32)
        self.wrongActions = np.random.random_integers(0, 1, self.totalState)
        self.params = np.random.normal(0, 0.1, (self.totalState * self.totalAction + 1)).astype(np.float32)
        
    def getAction(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 1) 
        else:
            return np.argmax(self.getQval(s))
    
    def getFeatures(self, s, a):
        features = np.zeros((self.totalState * self.totalAction + 1), dtype=np.int)
        features[s * self.totalAction + a] = 1                      
        features[self.totalState + self.totalAction] = 1       # bias
        return features
        
    def getQval(self, s, a=None):
        if self.mode == 'FA':
            if a == None:
                values = []
                values.append(self.params.dot(self.getFeatures(s, 0)))
                values.append(self.params.dot(self.getFeatures(s, 1)))
                return values
            else:
                return self.params.dot(self.getFeatures(s, a))
        else:
            if a == None:
                return self.Qval[s]
            else:
                return self.Qval[s, a]
        
    def isWrongAction(self, s, a):
        if self.wrongActions[s] == a:
            return True
        else:
            return False
        
    def doAction(self, s, a):
        """ returns next state and reward """
         
        if self.isWrongAction(s, a):
            return 0, 0
        else:
            if s < self.totalState - 1:
                return s+1, 0
            else:
                return 0, 1

    def updateValue(self, s, a, td):
        if self.mode == 'FA':
            self.params += self.stepSize * td * self.getFeatures(s, a)
        else:
            self.Qval[s, a] = self.getQval(s, a) + self.stepSize * td
            
    def isComplete(self):
        for s in range(self.totalState):
            policy = np.argmax(self.getQval(s))
            if self.isWrongAction(s, policy) == True:
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
                self.updateValue(s, a, td)
                s = s2
                
                if i % 10 == 0:
                    if self.isComplete():
                        #print 'training done %s out of %s' % (repeat, self.repeatNo)
                        trainDone.append(i)
                        break
        
        print '%s' % trainDone
        print '%s state training complete with %s. mean iters = %.0f' % (self.totalState, self.mode, np.mean(trainDone))
        
if __name__ == '__main__':
    Tester().gogo()