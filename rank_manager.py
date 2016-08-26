import random
from replay_memory import ReplayMemory
import numpy as np

class RankManager:
    def __init__(self, size, batchSize, historyLen, width, height):
        self.batchSize = batchSize
        self.historyLen = historyLen
        self.replayMemory = ReplayMemory(size, batchSize, historyLen, width, height)
        self.heapIndexList = [-1] * size        # This list maps replayIndex to heapIndex
        self.heap = []                                     # Binary heap
        self.heap.append((None, None))
        self.alpha = 0.7
        self.beta = 0.5
        self.maxWeight = 0
        self.newWeight = 100
        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((batchSize, historyLen, height, width), dtype = np.uint8)
        self.poststates = np.empty((batchSize, historyLen, height, width), dtype = np.uint8)
        self.rankSegmentIndex = {}              # heap indexes for each segment

    @property
    def count(self):
        return self.replayMemory.count

    def add(self, action, reward, screen, terminal, weight=None):
        if weight == None:
            weight = self.newWeight
        addedReplayIndex = self.replayMemory.add(action, reward, screen, terminal)

        # If there was the same data then remove it first
        heapIndex = self.heapIndexList[addedReplayIndex]
        if heapIndex != -1:
            self.remove(heapIndex)
        
        item = (addedReplayIndex, weight)
        self.heap.append(item)
        childIndex = len(self.heap) - 1
        self.heapIndexList[addedReplayIndex] = childIndex
        self.reorderUpward(childIndex)
        
    def remove(self, index):
        lastIndex = len(self.heap) - 1
        self.heapIndexList[self.heap[index][0]] = -1
        self.heap[index] = self.heap[lastIndex]
        self.heapIndexList[self.heap[index][0]] = index
        self.heap.pop(lastIndex)
        self.reorder(index)
        
    def getTop(self):
        return self.heap[1]
    
    def get(self, index):
        return self.heap[index]
    
    def getLength(self):
        return len(self.heap) - 1
    
    def swap(self, index1, item1, index2, item2):
        self.heap[index1] = item1
        self.heap[index2] = item2
        self.heapIndexList[item1[0]] = index1
        self.heapIndexList[item2[0]] = index2

    def reorder(self, index, newValue=None):
        if newValue != None:
            self.heap[index] = (self.heap[index][0], newValue)

        reordered = self.reorderUpward(index)
        
        if reordered == False:
            self.reorderDownward(index) 
        
    def reorderUpward(self, index):
        reordered = False
        childIndex = index

        while True:
            item = self.heap[childIndex]
            parentIndex = childIndex / 2
            parentItem = self.heap[parentIndex]
            if parentIndex == 0 or item[1] <= parentItem[1]:
                break
            self.swap(parentIndex, item, childIndex, parentItem)            
            childIndex = parentIndex
            reordered = True
            
        return reordered
        
    def reorderDownward(self, index):
        parentIndex = index
        while True:
            parentItem = self.heap[parentIndex]
            childIndex1 = parentIndex * 2
            childIndex2 = parentIndex * 2 + 1
            
            if childIndex2 > len(self.heap) - 1:
                if childIndex1 <= len(self.heap) - 1 and self.heap[childIndex1][1] > parentItem[1]:
                    self.swap(parentIndex, self.heap[childIndex1], childIndex1, parentItem)
                    self.heapIndexList[self.heap[childIndex1][0]] = parentIndex
                    self.heapIndexList[parentItem[0]] = childIndex1
                    
                    parentIndex = childIndex1
                else:
                    break
            else:
                if self.heap[childIndex1][1] > parentItem[1] and self.heap[childIndex1][1] >= self.heap[childIndex2][1]:
                    self.swap(parentIndex, self.heap[childIndex1], childIndex1, parentItem)
                    parentIndex = childIndex1
                elif self.heap[childIndex2][1] > parentItem[1] and self.heap[childIndex2][1] >= self.heap[childIndex1][1]:
                    self.swap(parentIndex, self.heap[childIndex2], childIndex2, parentItem)
                    parentIndex = childIndex2
                else:
                    break
            
    def reorderTop(self, newTopValue):
        top = self.getTop()
        self.heap[1] = (top[0], newTopValue) 
        
        self.reorderDownward(1) 

    def sort(self):
        newHeap = []
        newHeap.append((None, None))
        heapSize = len(self.heap)
        #print 'heapSize : %s' % heapSize
        for i in range(1, heapSize):
            #print 'i : %s' % i
            top = self.getTop()
            newHeap.append(top)
            lastIndex = heapSize - i
            #print 'lastIndex : %s' % lastIndex
            self.heap[1] = self.heap[lastIndex]
            self.heap.pop(lastIndex)
            if lastIndex > 1:
                self.reorderDownward(1)
            
        self.heap = newHeap        
        
        for i in range(1, heapSize):
            self.heapIndexList[self.heap[i][0]] = i
        
    def updateWeight(self, heapIndex, weight):
        self.reorder(heapIndex, weight)
    
    def getSegments(self):
        dataLen = len(self.heap) - 1
        segment = dataLen / 100000 * 100000
        if segment == 0:       # If data len is less than necessary size then use uniform segments
            return None
        else:
            if segment not in self.rankSegmentIndex:
                self.rankSegmentIndex[segment] = self.calculateSegments(segment)
            return self.rankSegmentIndex[segment]

    def calculateSegments(self, dataLen=None):
        if dataLen == None:
            dataLen = len(self.heap)
            
        rankSum = 0
        for i in range(dataLen):
            rankSum += (1.0 / (i + 1)) ** self.alpha

        segment = rankSum / self.batchSize
        segmentRankSum = 0
        segmentNo = 1
        rankSegmentIndex = []
        for i in range(1, dataLen):
            segmentRankSum += (1.0 / i) ** self.alpha
            if segmentRankSum >= segment * segmentNo:
                rankSegmentIndex.append(i)
                segmentNo += 1
        rankSegmentIndex.append(len(self.heap) - 1)
        
        """
        for i in range(len(self.heap)):
            print 'self.heap[%s] : %s, %s' % (i, self.heap[i][0], self.heap[i][1])
        print 'rankSegmentIndex : %s' % rankSegmentIndex
        """
        return rankSegmentIndex
                
    def getMinibatch(self):
        rankSegmentIndex = self.getSegments()
        
        # sample random indexes
        indexes = []
        heapIndexes = []
        weightList = []        
        for segment in range(self.batchSize):
            if rankSegmentIndex == None:
                    index1 = 1
                    index2 = self.count - 1
            else:
                if segment == 0:
                    index1 = 1
                else:
                    index1 = rankSegmentIndex[segment-1] + 1
                index2 = rankSegmentIndex[segment]

            # find index 
            while True:
                heapIndex = random.randint(index1, index2)
                replayIndex = self.heap[heapIndex][0]
                
                repeatAgain = False
                if replayIndex < self.historyLen:
                    repeatAgain = True
                # if wraps over current pointer, then get new one
                if replayIndex >= self.replayMemory.current and replayIndex - self.historyLen < self.replayMemory.current:
                    repeatAgain = True
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.replayMemory.terminals[(replayIndex - self.historyLen):replayIndex].any():
                    repeatAgain = True
                
                if repeatAgain:
                    index1 = 1
                    index2 = self.count - 1
                    continue            

                # otherwise use this index
                weight = (1.0 / heapIndex / len(self.heap)) ** self.beta
                if weight > self.maxWeight:
                    self.maxWeight = weight
                
                weight = weight / self.maxWeight
                weightList.append(weight)
                break
                
            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.replayMemory.getState(replayIndex - 1)
            self.poststates[len(indexes), ...] = self.replayMemory.getState(replayIndex)
            indexes.append(replayIndex)
            heapIndexes.append(heapIndex)
    
        # copy actions, rewards and terminals with direct slicing
        actions = self.replayMemory.actions[indexes]
        rewards = self.replayMemory.rewards[indexes]
        terminals = self.replayMemory.terminals[indexes]
        return self.prestates, actions, rewards, self.poststates, terminals, indexes, heapIndexes, weightList

        