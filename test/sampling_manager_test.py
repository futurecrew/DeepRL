import sys
import random
import time
import numpy as np
import unittest
from sampling_manager import SamplingManager

class TestSamplingManager(unittest.TestCase):

#class TestSamplingManager:
#    def __init__(self):
#        self.initialize()
        
    def setUp(self):
        self.initialize()
        
    def initialize(self):
        #self.mode = 'PROPORTION'
        self.mode = 'RANK'
        
        if self.mode == 'PROPORTION':
            self.alpha = 0.6
            self.beta = 0.4
            self.sortTerm = 250000
        elif self.mode == 'RANK':
            self.alpha = 0.7
            self.beta = 0.5
            self.sortTerm = 250000

    def checkHeapIndexListValidity(self, manager):
        for i in range(1, len(manager.heap)):
            replayIndex = manager.heap[i][0]
            self.assertEquals(manager.heapIndexList[replayIndex], i)

    def test_Add(self):
        replayMemorySize = 3
        totalList = [5, 10, 15, 100]
        
        for dataLen in totalList:
            #print 'dataLen : %s' % dataLen
            manager = SamplingManager(replayMemorySize, 32, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0)

            dataLen2 = min(dataLen, replayMemorySize)            
            self.assertEqual(manager.getHeapLength(), dataLen2)
            self.assertEqual(manager.count, dataLen2)
                
            for replayIndex in range(dataLen2):
                heapIndex = manager.heapIndexList[replayIndex]
                heapItem = manager.get(heapIndex)
                self.assertEqual(replayIndex, heapItem[0])
    
    def atest_Add2(self):
        replayMemorySize = 1000000
        
        for t in range(2):
            manager = SamplingManager(replayMemorySize, 32, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
            for i in range(2200000):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                if t == 0:
                    manager.add(action=0, reward=0, screen=state, terminal=0)
                else:
                    manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            self.assertEqual(manager.count, replayMemorySize)
            self.assertEqual(manager.getHeapLength(), replayMemorySize)

    def test_GetMinibatch(self):    
        replayMemorySize = 100000
        dataLen = 220000
        minibatchSize = 32
        manager = SamplingManager(replayMemorySize, minibatchSize, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
        for i in range(dataLen):
            state = np.zeros((84, 84), dtype=np.int)
            state.fill(i)
            manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
        pres, actions, rewards, posts, terminals, replayIndexes, heapIndexes, weights = manager.getMinibatch()
        self.assertEqual(minibatchSize, len(actions))
        
        # Weights should be ascending order
        if self.mode == 'RANK':
            error = False
            prevWeight = 0
            for weight in weights:
                if weight < prevWeight:
                    error = True
                    break
                prevWeight = weight
            self.assertEquals(error, False)
        
        #print 'heapIndexList : %s' % manager.heapIndexList
        #print 'heapIndexes : %s' % heapIndexes
        print 'replayIndexes : %s' % replayIndexes
        
        manager.sort()

        pres, actions, rewards, posts, terminals, replayIndexes, heapIndexes, weights = manager.getMinibatch()
        
        print 'replayIndexes : %s' % replayIndexes
    
    def atest_GetMinibatch2(self):    
        replayMemorySize = 100000
        minibatchSize = 32
        dataLenList = [1000, 100000, 220000]
        for dataLen in dataLenList:
            manager = SamplingManager(replayMemorySize, minibatchSize, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=100)
    
            memorySizeToCheck = min(replayMemorySize, dataLen)
            startIndex = dataLen % replayMemorySize
            visited = {}
            for i in range(memorySizeToCheck):
                visited[i] = 0
                
            for i in range(replayMemorySize):
                pres, actions, rewards, posts, terminals, replayIndexes, heapIndexes, weights = manager.getMinibatch()
                for replayIndex in replayIndexes:
                    visited[replayIndex] += 1
                
            everyIndexVisited = True
            for i in range(4, memorySizeToCheck):
                if visited[i] == 0 and (i < startIndex or i > startIndex + 3):
                    print 'index %s is not visited' % i
                    everyIndexVisited = False
                    #manager.getMinibatch()
                    #break
            
            #self.assertEqual(everyIndexVisited, True)
                    
        
    def atest_GetMinibatch3(self):    
        replayMemorySize = 100000
        minibatchSize = 32
        dataLenList = [1000, 100000, 200000]
        for dataLen in dataLenList:
            manager = SamplingManager(replayMemorySize, minibatchSize, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            manager.sort()
            
            memorySizeToCheck = min(replayMemorySize, dataLen)
            startIndex = dataLen % replayMemorySize
            visited = {}
            for i in range(memorySizeToCheck):
                visited[i] = 0
                
            for i in range(replayMemorySize):
                pres, actions, rewards, posts, terminals, replayIndexes, heapIndexes, weights = manager.getMinibatch()
                for replayIndex in replayIndexes:
                    visited[replayIndex] += 1
            
            if dataLen >= 100000:
                for i in range(4, memorySizeToCheck / 10):
                    self.assertGreater(visited[memorySizeToCheck-i-1], visited[i])
                    
    
    def atest_Sort(self):
        replayMemorySize = 10**6
        dataLenList = [100, 1000, 2000, 10**6]
        for dataLen in dataLenList:
            minibatchSize = 32
            manager = SamplingManager(replayMemorySize, minibatchSize, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            #for i in range(len(manager.heap)):
            #    print 'heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])

            self.checkHeapIndexListValidity(manager)
            
            startTime = time.time()
            manager.sort()
            
            print 'sort() %s data took %.1f sec' % (dataLen, time.time() - startTime)

            self.checkHeapIndexListValidity(manager)
                
            prevTD = sys.maxint
            for i in range(1, len(manager.heap)):
                td = manager.heap[i][1]
                self.assertGreaterEqual(prevTD, td)
                prevTD = td
            
            #for i in range(len(manager.heap)):
            #    print 'heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])
        
    def test_UpdateTD(self):
        replayMemorySize = 1000
        dataLenList = [100, 1000, 2000, 2200]
        for dataLen in dataLenList:
            minibatchSize = 32
            manager = SamplingManager(replayMemorySize, minibatchSize, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            self.checkHeapIndexListValidity(manager)

            # Change td of the top item to the lowest then check the item is in the last
            heapIndex = 1
            item = manager.heap[heapIndex]
            replayIndex = item[0]
            newTD = -sys.maxint
            manager.updateTD(heapIndex, newTD)

            item2 = manager.heap[len(manager.heap) - 1]
            replayIndex2 = item2[0]
            
            self.assertEqual(replayIndex, replayIndex2)
            
            # Increase td of an item then check the item is in the higher index
            for i in range(100):        # Test 100 times
                heapIndex = random.randint(1, len(manager.heap) - 1)
                item = manager.heap[heapIndex]
                replayIndex = item[0]
                td = item[1]
                newTD = td + 10
                manager.updateTD(heapIndex, newTD)
    
                newHeapIndex = manager.heapIndexList[replayIndex]
                item2 = manager.heap[newHeapIndex]
                
                self.assertEquals(item[0], item2[0])
                self.assertGreaterEqual(heapIndex, newHeapIndex)

            self.checkHeapIndexListValidity(manager)

    def test_CalculateSegments(self):
        replayMemorySize = 10**6
        #dataLenList = [50000, 100000, 1000000, 2000000]
        #dataLenList = [1000000]
        dataLenList = [10000]
        startTD = 0.000001
        endTD = 1.0
        for dataLen in dataLenList:
            
            print 'testing %s' % dataLen
            
            tdIncrease = (endTD - startTD) / dataLen
            minibatchSize = 32
            manager = SamplingManager(replayMemorySize, minibatchSize, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
            
            time1 = time.time()
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                #manager.add(action=0, reward=0, screen=state, terminal=0, td=startTD + i * tdIncrease)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            time2 = time.time()
            manager.calculateSegments()
            time3 = time.time()
            
            print 'Adding %d took %.1f sec.' % (dataLen, time2 - time1)
            print 'Calculating segments took %.1f sec.' % (time3 - time2)

    def test_GetSegments(self):
        #dataLen = 10**6
        dataLen = 10**4
        replayMemorySize = dataLen
        minibatchSize = 32
        startTD = 0.000001
        endTD = 1.0
        tdIncrease = (endTD - startTD) / dataLen
        manager = SamplingManager(replayMemorySize, minibatchSize, 4, 84, 84, self.mode, self.alpha, self.beta, self.sortTerm)
        for i in range(dataLen):
            state = np.zeros((84, 84), dtype=np.int)
            state.fill(i)
            #manager.add(action=0, reward=0, screen=state, terminal=0, td=startTD + i * tdIncrease)
            manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
            segmentIndex = manager.getSegments()
            
            if i % 100000 == 0 or i == dataLen - 1:
                print 'segmentIndex : %s' % segmentIndex

        for i in range(len(manager.heap)):
            print 'manager.heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])
            
        manager.sort()
        segmentIndex = manager.getSegments()
        print 'segmentIndex : %s' % segmentIndex
        
if __name__ == '__main__':
    unittest.main()
    #TestSamplingManager().test_GetSegments()
    #TestSamplingManager().test_CalculateSegments()
