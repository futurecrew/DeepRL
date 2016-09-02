import numpy as np
import unittest
from binary_heap import BinaryHeap
from replay_memory import ReplayMemory

class TestBinaryHeap(unittest.TestCase):
    def setUp(self):
        self.heap = BinaryHeap()
        self.replayMemory = ReplayMemory(10, 32, 4, 84, 84)

    def test_Add(self):
        totalNo = 10
        for i in range(totalNo):
            state = np.zeros((84, 84), dtype=np.int)
            state.fill(i)
            td = i
            
            addedIndex = self.replayMemory.add(0, 0, state, 0)
            self.heap.add(addedIndex, td)
            
        for i in range(totalNo):
            topItem = self.heap.getTop()
            self.assertEqual(totalNo - i - 1, topItem[0])
            self.heap.remove(0)
        
if __name__ == '__main__':
    unittest.main()