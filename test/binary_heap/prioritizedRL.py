import random
import time
import numpy as np
from binaryHeap import BinaryHeap

class Tester:
    def __init__(self):
        self.totalState = 10
        self.totalAction = 2
        self.epsilon = 0.1 
        self.stepSize = 0.25
        #self.stepSize = 0.05
        self.discount = 1.0 - 1.0 / self.totalState
        self.maxIter = 10**6
        self.repeatNo = 10
        self.minibatch = 5
        self.alpha = 0.7
        self.beta = 0.5
        self.replayMemory = []
        self.maxWeight = 0
        
        self.mode = 'FA'
        #self.mode = 'Tablar'
        
        #self.samplePolicy = 'uniform'
        #self.samplePolicy = 'maxPriority'
        self.samplePolicy = 'rank'
        self.binaryHeap = BinaryHeap()
        
        print 'replay memory size : %s' % (2**(self.totalState + 1) - 2)
        
    def initialize(self):
        self.Qval = np.random.normal(0, 0.1, (self.totalState, self.totalAction)).astype(np.float32)
        self.params = np.random.normal(0, 0.1, (self.totalState * self.totalAction + 1)).astype(np.float32)
        self.wrongActions = np.zeros((self.totalState), dtype=np.int8)
        for i in range(self.totalAction):
            if i % 2 == 0:
                self.wrongActions[i] = 1
        self.replayMemory = []
        self.generateReplay()
    
    def generateReplay(self):
        for s in range(self.totalState-1, -1, -1):
            repeat = 2 ** (self.totalState - s - 1)
            for r in range(repeat):
                a = self.getTrueAction(s)
                s2, r = self.doAction(s, a)
                self.replayMemory.append((s, a, r, s2))

                a = self.getWrongAction(s)
                s2, r = self.doAction(s, a)
                self.replayMemory.append((s, a, r, s2))
        random.shuffle(self.replayMemory)

        # Generate binary heap
        rank = 1
        rankSum = 0
        for data in self.replayMemory:
            self.binaryHeap.add(data, 1.0)
            rankSum += (1.0 / rank) ** self.alpha
            rank += 1
        
        self.rankIndex = []
        segment = rankSum / self.minibatch 
        if self.samplePolicy == 'rank':
            rank = 1
            segmentRankSum = 0
            segmentIndex = 0
            for i in range(1, len(self.replayMemory)):
                segmentRankSum += (1.0 / rank) ** self.alpha
                rank += 1
                if segmentRankSum >= segment:
                    self.rankIndex.append(i)
                    segmentIndex += 1
                    segmentRankSum = 0
            self.rankIndex.append(len(self.replayMemory) - 1)
        
    def getAction(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 1) 
        else:
            return np.argmax(self.getQval(s))
    
    def getFeatures(self, s, a):
        features = np.zeros((self.totalState * self.totalAction + 1), dtype=np.int)
        features[s * self.totalAction + a] = 1                      
        features[self.totalState * self.totalAction] = 1       # bias
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
        
    def getTrueAction(self, s):
        return 1 - self.wrongActions[s]
        
    def getWrongAction(self, s):
        return self.wrongActions[s]
        
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

        if self.samplePolicy == 'maxPriority':
            self.binaryHeap.reorderTop(np.abs(td))
            
    def isComplete(self):
        R = 1.0
        error = 0
        for s in range(self.totalState-1, -1, -1):
            for a in range(self.totalAction):
                estimate = self.getQval(s, a)
                if self.isWrongAction(s, a):
                    groundTruth = 0
                else:
                    groundTruth = R
                error += np.square(groundTruth - estimate)
            R *= self.discount
            
        if error / (self.totalState * self.totalAction) <= 10**-3:
            return True
        else:
            return False

    def sampleReplay(self, segment):
        if self.samplePolicy == 'uniform':
            index = random.randint(0, len(self.replayMemory)-1)
            return index, 1.0, self.replayMemory[index]
        elif self.samplePolicy == 'maxPriority':
            index = 1
            item = self.binaryHeap.getTop()
            return index, 1.0, item[0]
        elif self.samplePolicy == 'rank':
            if segment == 0:
                index1 = 1
            else:
                index1 = self.rankIndex[segment-1] + 1
            index2 = self.rankIndex[segment]
            index = random.randint(index1, index2)
            item = self.binaryHeap.heap[index]
            weight = (1.0 / index / self.totalState) ** self.beta
            
            if weight > self.maxWeight:
                self.maxWeight = weight
            
            weight = weight / self.maxWeight
            # DJDJ
            #weight = 1.0
            return index, weight, item[0]

    def gogoReplay(self):
        print 'Training replay : policy %s' % self.samplePolicy

        startTime = time.time()
        trainDone = []
        paramSum = np.zeros((self.totalState * self.totalAction + 1))
        for repeat in range(self.repeatNo):
            self.initialize()
            
            for i in range(self.maxIter):
                paramSum.fill(0)
                for m in range(self.minibatch):
                    index, weight, (s, a, r, s2) = self.sampleReplay(m)
                    if s2 == 0:     # terminal state
                        td = r - self.getQval(s, a)
                    else:
                        td = r + self.discount * np.max(self.getQval(s2)) - self.getQval(s, a)

                    if self.mode == 'FA':
                        paramSum += self.stepSize * weight * td * self.getFeatures(s, a)
                    else:
                        self.Qval[s, a] = self.getQval(s, a) + self.stepSize * td
            
                    self.binaryHeap.reorder(index, np.abs(td))

                self.params += paramSum

                if i % 10 == 0:
                    if self.isComplete():
                        print 'training done %s out of %s' % (repeat+1, self.repeatNo)
                        trainDone.append(i)
                        break
        
        print '%s' % trainDone
        print '%s state training complete with %s mean iters = %.0f' % (self.totalState, self.mode, np.mean(trainDone))
        print 'elapsed : %.1fs' % (time.time() - startTime)
        
    def gogoOnline(self):
        print 'Training online'

        trainDone = []
        for repeat in range(self.repeatNo):
            s = 0
            self.initialize()
            
            for i in range(self.maxIter):
                a = self.getAction(s)
                s2, r = self.doAction(s, a)
                if s2 == 0:     # terminal state
                    td = r - self.getQval(s, a)
                else:
                    td = r + self.discount * np.max(self.getQval(s2)) - self.getQval(s, a)
                self.updateValue(s, a, td)
                s = s2
                
                if i % 10 == 0:
                    if self.isComplete():
                        #print 'training done %s out of %s' % (repeat+1, self.repeatNo)
                        trainDone.append(i)
                        break
        
        print '%s' % trainDone
        print '%s state training complete with %s mean iters = %.0f' % (self.totalState, self.mode, np.mean(trainDone))
        
if __name__ == '__main__':
    #Tester().gogoOnline()
    Tester().gogoReplay()