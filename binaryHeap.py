class BinaryHeap:
    def __init__(self):
        self.heap = []
        self.heap.append((None, None))
    
    def add(self, data, value):
        item = (data, value)
        self.heap.append(item)
        childIndex = len(self.heap) - 1

        while True:
            parentIndex = childIndex / 2
            parentItem = self.heap[parentIndex]
            if parentIndex == 0 or value <= parentItem[1]:
                break
            self.heap[parentIndex] = item
            self.heap[childIndex] = parentItem
            childIndex = parentIndex
            
    def getTop(self):
        return self.heap[1]
    
    def reorderTop(self, newTopValue):
        top = self.getTop()
        self.heap[1] = (top[0], newTopValue) 
        
        parentIndex = 1
        while True:
            parentItem = self.heap[parentIndex]
            childIndex1 = parentIndex * 2
            childIndex2 = parentIndex * 2 + 1
            
            if childIndex2 > len(self.heap) - 1:
                if childIndex1 <= len(self.heap) - 1 and self.heap[childIndex1][1] > parentItem[1]:
                    self.heap[parentIndex] = self.heap[childIndex1]
                    self.heap[childIndex1] = parentItem
                    parentIndex = childIndex1
                else:
                    break
            else:
                if self.heap[childIndex1][1] > parentItem[1] and self.heap[childIndex1][1] >= self.heap[childIndex2][1]:
                    self.heap[parentIndex] = self.heap[childIndex1]
                    self.heap[childIndex1] = parentItem
                    parentIndex = childIndex1
                elif self.heap[childIndex2][1] > parentItem[1] and self.heap[childIndex2][1] >= self.heap[childIndex1][1]:
                    self.heap[parentIndex] = self.heap[childIndex2]
                    self.heap[childIndex2] = parentItem
                    parentIndex = childIndex2
                else:
                    break
        