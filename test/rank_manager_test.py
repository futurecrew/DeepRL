import sys
import random
import time
import numpy as np
import unittest
from rank_manager import RankManager

class TestRankManager(unittest.TestCase):
    def setUp(self):
        self.replayMemorySize = 3

    def checkHeapIndexListValidity(self, manager):
        for i in range(1, len(manager.heap)):
            replayIndex = manager.heap[i][0]
            self.assertEquals(manager.heapIndexList[replayIndex], i)

    def test_Add(self):
        totalList = [5, 10, 15, 100]
        
        for dataLen in totalList:
            #print 'dataLen : %s' % dataLen
            manager = RankManager(self.replayMemorySize, 32, 4, 84, 84)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0)

            dataLen2 = min(dataLen, self.replayMemorySize)            
            self.assertEqual(manager.getLength(), dataLen2)
            self.assertEqual(manager.count, dataLen2)
                
            for replayIndex in range(dataLen2):
                heapIndex = manager.heapIndexList[replayIndex]
                heapItem = manager.get(heapIndex)
                self.assertEqual(replayIndex, heapItem[0])
    
    def test_GetMinibatch(self):    
        replayMemorySize = 100000
        dataLen = 220000
        minibatchSize = 32
        manager = RankManager(replayMemorySize, minibatchSize, 4, 84, 84)
        for i in range(dataLen):
            state = np.zeros((84, 84), dtype=np.int)
            state.fill(i)
            manager.add(action=0, reward=0, screen=state, terminal=0, weight=i)
    
        pres, actions, rewards, posts, terminals, replayIndexes, heapIndexes, weightList = manager.getMinibatch()
        self.assertEqual(minibatchSize, len(actions))
        
        #print 'heapIndexList : %s' % manager.heapIndexList
        #print 'heapIndexes : %s' % heapIndexes
        print 'replayIndexes : %s' % replayIndexes
        
        manager.sort()

        pres, actions, rewards, posts, terminals, replayIndexes, heapIndexes, weightList = manager.getMinibatch()
        
        print 'replayIndexes : %s' % replayIndexes
    
    def test_Sort(self):
        replayMemorySize = 10**6
        dataLenList = [100, 1000, 2000, 10**6]
        for dataLen in dataLenList:
            minibatchSize = 32
            manager = RankManager(replayMemorySize, minibatchSize, 4, 84, 84)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, weight=i)
    
            #for i in range(len(manager.heap)):
            #    print 'heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])

            self.checkHeapIndexListValidity(manager)
                
            manager.sort()

            self.checkHeapIndexListValidity(manager)
                
            prevWeight = sys.maxint
            for i in range(1, len(manager.heap)):
                weight = manager.heap[i][1]
                self.assertGreaterEqual(prevWeight, weight)
                prevWeight = weight
            
            #for i in range(len(manager.heap)):
            #    print 'heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])
        
    def test_UpdateWeights(self):
        replayMemorySize = 1000
        dataLenList = [100, 1000, 2000, 2200]
        for dataLen in dataLenList:
            minibatchSize = 32
            manager = RankManager(replayMemorySize, minibatchSize, 4, 84, 84)
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, weight=i)
    
            self.checkHeapIndexListValidity(manager)

            # Change weight of the top item to the lowest then check the item is in the last
            heapIndex = 1
            item = manager.heap[heapIndex]
            replayIndex = item[0]
            newWeight = -sys.maxint
            manager.updateWeight(heapIndex, newWeight)

            item2 = manager.heap[len(manager.heap) - 1]
            replayIndex2 = item2[0]
            
            self.assertEqual(replayIndex, replayIndex2)
            
            # Increase weight of an item then check the item is in the higher index
            for i in range(100):        # Test 100 times
                heapIndex = random.randint(1, len(manager.heap) - 1)
                item = manager.heap[heapIndex]
                replayIndex = item[0]
                weight = item[1]
                newWeight = weight + 10
                manager.updateWeight(heapIndex, newWeight)
    
                newHeapIndex = manager.heapIndexList[replayIndex]
                item2 = manager.heap[newHeapIndex]
                
                self.assertEquals(item[0], item2[0])
                self.assertGreaterEqual(heapIndex, newHeapIndex)

            self.checkHeapIndexListValidity(manager)

    def atest_CalculateSegments(self):
        replayMemorySize = 10**6
        dataLenList = [50000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
        for dataLen in dataLenList:
            minibatchSize = 32
            manager = RankManager(replayMemorySize, minibatchSize, 4, 84, 84)
            
            time1 = time.time()
            for i in range(dataLen):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, weight=i)
    
            time2 = time.time()
            manager.calculateSegments()
            time3 = time.time()
            
            print 'Adding %d took %.1f sec.' % (dataLen, time2 - time1)
            print 'Calculating segments took %.1f sec.' % (time3 - time2)

    def test_GetSegments(self):
        replayMemorySize = 10**6
        minibatchSize = 32
        manager = RankManager(replayMemorySize, minibatchSize, 4, 84, 84)
        for i in range(10**6):
            state = np.zeros((84, 84), dtype=np.int)
            state.fill(i)
            manager.add(action=0, reward=0, screen=state, terminal=0, weight=i)
            segmentIndex = manager.getSegments()
            
            if i % 100000 == 0:
                print 'segmentIndex : %s' % segmentIndex
            
        
if __name__ == '__main__':
    unittest.main()