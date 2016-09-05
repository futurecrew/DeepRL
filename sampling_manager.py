import random
from replay_memory import ReplayMemory
import numpy as np

class SamplingManager:
    def __init__(self, replayMemory, useGpuReplayMem, size, batchSize, historyLen, samplingMode,
                            samplingAlpha, samplingBeta, sortTerm):
        self.replayMemory = replayMemory
        self.useGpuReplayMem = useGpuReplayMem
        self.batchSize = batchSize
        self.historyLen = historyLen
        self.samplingMode = samplingMode
        self.alpha = samplingAlpha
        self.beta = samplingBeta
        self.sortTerm = sortTerm
        self.heapIndexList = [-1] * size        # This list maps replayIndex to heapIndex
        self.heap = []                                     # Binary heap
        self.heap.append((None, None))
        self.proportionEpsilon = 0.0000001
        self.maxWeight= 0
        self.maxTD = 1.0
        self.segmentCalculationUnit = 1000
        self.segmentIndex = {}              # heap indexes for each segment
        self.addCallNo = 0

    @property
    def count(self):
        return self.replayMemory.count

    def add(self, action, reward, screen, terminal, td=None):
        if td == None:
            td = self.maxTD
        addedReplayIndex = self.replayMemory.add(action, reward, screen, terminal)

        # If there was the same data then remove it first
        heapIndex = self.heapIndexList[addedReplayIndex]
        if heapIndex != -1:
            self.remove(heapIndex)
        
        item = (addedReplayIndex, td)
        self.heap.append(item)
        childIndex = len(self.heap) - 1
        self.heapIndexList[addedReplayIndex] = childIndex
        self.reorderUpward(childIndex)
        
        self.addCallNo += 1
        
        if self.addCallNo % self.sortTerm == 0:
            self.sort()
        if self.addCallNo % (10**5) == 0:     # Clear segmentIndex to calculate segment again
            self.segmentIndex = {}
        
    def remove(self, index):
        lastIndex = len(self.heap) - 1
        self.heapIndexList[self.heap[index][0]] = -1
        if index == lastIndex:
            self.heap.pop(lastIndex)
        else:
            self.heap[index] = self.heap[lastIndex]
            self.heapIndexList[self.heap[index][0]] = index
            self.heap.pop(lastIndex)
            self.reorder(index)
        
    def getTop(self):
        return self.heap[1]
    
    def get(self, index):
        return self.heap[index]
    
    def getHeapLength(self):
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
            
        self.segmentIndex = {}
        
    def updateTD(self, heapIndex, td):
        if td > self.maxTD:
            self.maxTD = td
        self.reorder(heapIndex, td)
    
    def getSegments(self):
        dataLen = len(self.heap) - 1
        segment = dataLen / self.segmentCalculationUnit * self.segmentCalculationUnit
        if segment == 0:       # If data len is less than necessary size then use uniform segments
            return None
        else:
            if segment not in self.segmentIndex:
                self.segmentIndex[segment] = self.calculateSegments(segment)
            return self.segmentIndex[segment]

    def getP(self, heapIndex):
        if self.samplingMode == 'RANK':
            return (1.0 / heapIndex) ** self.alpha
        elif self.samplingMode == 'PROPORTION':
            return (abs(self.heap[heapIndex][1]) + self.proportionEpsilon) ** self.alpha
        
    def calculateSegments(self, dataLen=None):
        if dataLen == None:
            dataLen = len(self.heap)
            
        self.totalPSum = 0
        for i in range(1, dataLen):
            self.totalPSum += self.getP(i)

        segment = self.totalPSum / self.batchSize
        segmentSum = 0
        segmentNo = 1
        segmentIndex = []
        for i in range(1, dataLen):
            segmentSum += self.getP(i)
            if segmentSum >= segment * segmentNo:
                segmentIndex.append(i)
                segmentNo += 1
                if len(segmentIndex) == self.batchSize - 1:
                    segmentIndex.append(len(self.heap) - 1)
                    break
        
        """
        for i in range(len(self.heap)):
            print 'self.heap[%s] : %s, %s' % (i, self.heap[i][0], self.heap[i][1])
        print 'segmentIndex : %s' % segmentIndex
        """
        return segmentIndex
            
    def getMinibatch(self):
        segmentIndex = self.getSegments()
        
        # sample random indexes
        indexes = []
        heapIndexes = []
        weights = []        
        for segment in range(self.batchSize):
            if segmentIndex == None:
                    index1 = 1
                    index2 = self.count
            else:
                if segment == 0:
                    index1 = 1
                else:
                    index1 = segmentIndex[segment-1] + 1
                index2 = segmentIndex[segment]

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
                    self.reorder(heapIndex, 0)         # Discard and never use this data again
                    continue            

                if segmentIndex == None:
                    weight = 1.0
                else:
                    weight = (self.totalPSum / self.getP(heapIndex) / len(self.heap)) ** self.beta
                    if weight > self.maxWeight:
                        self.maxWeight = weight
                    weight = weight / self.maxWeight
                weights.append(weight)
                break
                
            # NB! having index first is fastest in C-order matrices
            if self.useGpuReplayMem:
                self.replayMemory.prestates_view[len(indexes)][:] = self.replayMemory.getState(replayIndex - 1)
                self.replayMemory.poststates_view[len(indexes)][:] = self.replayMemory.getState(replayIndex)
            else:            
                self.replayMemory.prestates[len(indexes), ...] = self.replayMemory.getState(replayIndex - 1)
                self.replayMemory.poststates[len(indexes), ...] = self.replayMemory.getState(replayIndex)
            indexes.append(replayIndex)
            heapIndexes.append(heapIndex)
    
        # copy actions, rewards and terminals with direct slicing
        actions = self.replayMemory.actions[indexes]
        rewards = self.replayMemory.rewards[indexes]
        terminals = self.replayMemory.terminals[indexes]
        return self.replayMemory.prestates, actions, rewards, self.replayMemory.poststates, terminals, indexes, heapIndexes, weights

        