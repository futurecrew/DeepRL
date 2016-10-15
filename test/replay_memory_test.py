import unittest
import numpy as np
from replay_memory import ReplayMemory

class ReplayMemoryTest(unittest.TestCase):
    def setUp(self):
        self.use_gpu_replay_mem = False
        self.max_replay_memory = 100 
        self.train_batch_size = 5
        self.screen_history = 4
        self.screen_width = 3
        self.screen_height = 3
        self.minibatch_random = False
        #self.screen_order = 'shw'
        self.screen_order = 'hws'

    def test_get_minibatch(self):                                     
        replay_memory = ReplayMemory(None,
                                         self.use_gpu_replay_mem,
                                         self.max_replay_memory, 
                                         self.train_batch_size,
                                         self.screen_history,
                                         self.screen_width,
                                         self.screen_height,
                                         self.minibatch_random,
                                         self.screen_order)
        
        for i in range(255):
            screen = np.zeros((self.screen_height, self.screen_width))
            screen.fill(i + 1)
            replay_memory.add(i + 1, 10 * (i + 1), screen, False)
        
            if i > self.train_batch_size + self.screen_history:
                prestates, actions, rewards, poststates, terminals = replay_memory.get_minibatch()
                for b in range(self.train_batch_size-1):
                    for h in range(self.screen_history-1):
                        self.assertTrue(prestates[b+1, 0, 0, h] < prestates[b, 0, 0, h])
                        self.assertTrue(prestates[b, 0, 0, h+1] > prestates[b, 0, 0, h])
                        
                        
if __name__ == '__main__':
    unittest.main()
            