class BinaryHeap:
    def __init__(self):
        self.heap = []
        self.heap.append((None, None))
    
    def add(self, data, value):
        item = (data, value)
        self.heap.append(item)
        childIndex = len(self.heap) - 1
        
        reordered, newIndex = self.reorderUpward(childIndex)
        return newIndex
    
    def remove(self, index):
        lastIndex = len(self.heap) - 1
        self.heap[index] = self.heap[lastIndex]
        self.heap.pop(lastIndex)
        self.reorder(index)
        
    def getTop(self):
        return self.heap[1]
    
    def get(self, index):
        return self.heap[index]
    
    def getLength(self):
        return len(self.heap) - 1
    
    def reorder(self, index, newValue=None):
        if newValue != None:
            self.heap[index] = (self.heap[index][0], newValue)

        reordered, newIndex = self.reorderUpward(index)
        
        if reordered == False:
            self.reorderDownward(index) 
        
    def reorderUpward(self, index):
        reordered = False
        childIndex = index
        itemIndex = index

        while True:
            item = self.heap[childIndex]
            parentIndex = childIndex / 2
            parentItem = self.heap[parentIndex]
            if parentIndex == 0 or item[1] <= parentItem[1]:
                break
            self.heap[parentIndex] = item
            self.heap[childIndex] = parentItem
            childIndex = parentIndex
            itemIndex = parentIndex
            reordered = True
            
        return reordered, itemIndex
        
    def reorderDownward(self, index):
        parentIndex = index
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
            
    def reorderTop(self, newTopValue):
        top = self.getTop()
        self.heap[1] = (top[0], newTopValue) 
        
        self.reorderDownward(1) 

        