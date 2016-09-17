import sys
import random
import time
import numpy as np
import unittest
from replay_memory import ReplayMemory
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
            self.sort_term = 250000
        elif self.mode == 'RANK':
            self.alpha = 0.7
            self.beta = 0.5
            self.sort_term = 250000

    def check_heap_index_list_validity(self, manager):
        for i in range(1, len(manager.heap)):
            replay_index = manager.heap[i][0]
            self.assertEquals(manager.heap_index_list[replay_index], i)

    def test_add(self):
        replay_memory_size = 3
        totalList = [5, 10, 15, 100]
        
        for data_len in totalList:
            #print 'data_len : %s' % data_len
            uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
            manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, 32, 4, self.mode, self.alpha, self.beta, self.sort_term)
            for i in range(data_len):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0)

            data_len2 = min(data_len, replay_memory_size)            
            self.assertEqual(manager.get_heap_length(), data_len2)
            self.assertEqual(manager.count, data_len2)
                
            for replay_index in range(data_len2):
                heap_index = manager.heap_index_list[replay_index]
                heap_item = manager.get(heap_index)
                self.assertEqual(replay_index, heap_item[0])
    
    def atest_add2(self):
        replay_memory_size = 1000000
        
        for t in range(2):
            uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
            manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, 32, 4, self.mode, self.alpha, self.beta, self.sort_term)
            for i in range(2200000):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                if t == 0:
                    manager.add(action=0, reward=0, screen=state, terminal=0)
                else:
                    manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            self.assertEqual(manager.count, replay_memory_size)
            self.assertEqual(manager.get_heap_length(), replay_memory_size)

    def test_get_minibatch(self):    
        replay_memory_size = 100000
        data_len = 220000
        minibatch_size = 32
        uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
        manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, minibatch_size, 4, self.mode, self.alpha, self.beta, self.sort_term)
        for i in range(data_len):
            state = np.zeros((84, 84), dtype=np.int)
            state.fill(i)
            manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
        pres, actions, rewards, posts, terminals, replay_indexes, heap_indexes, weights = manager.get_minibatch()
        self.assertEqual(minibatch_size, len(actions))
        
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
        
        #print 'heap_index_list : %s' % manager.heap_index_list
        #print 'heap_indexes : %s' % heap_indexes
        print 'replay_indexes : %s' % replay_indexes
        
        manager.sort()

        pres, actions, rewards, posts, terminals, replay_indexes, heap_indexes, weights = manager.get_minibatch()
        
        print 'replay_indexes : %s' % replay_indexes
    
    def atest_get_minibatch2(self):    
        replay_memory_size = 100000
        minibatch_size = 32
        data_len_list = [1000, 100000, 220000]
        for data_len in data_len_list:
            uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
            manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, minibatch_size, 4, self.mode, self.alpha, self.beta, self.sort_term)
            for i in range(data_len):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=100)
    
            memory_size_to_check = min(replay_memory_size, data_len)
            start_index = data_len % replay_memory_size
            visited = {}
            for i in range(memory_size_to_check):
                visited[i] = 0
                
            for i in range(replay_memory_size):
                pres, actions, rewards, posts, terminals, replay_indexes, heap_indexes, weights = manager.get_minibatch()
                for replay_index in replay_indexes:
                    visited[replay_index] += 1
                
            everyIndexVisited = True
            for i in range(4, memory_size_to_check):
                if visited[i] == 0 and (i < start_index or i > start_index + 3):
                    print 'index %s is not visited' % i
                    everyIndexVisited = False
                    #manager.get_minibatch()
                    #break
            
            #self.assertEqual(everyIndexVisited, True)
                    
        
    def test_get_minibatch3(self):    
        replay_memory_size = 100000
        minibatch_size = 32
        data_len_list = [1000, 100000, 200000]
        for data_len in data_len_list:
            uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
            manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, minibatch_size, 4, self.mode, self.alpha, self.beta, self.sort_term)
            for i in range(data_len):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            manager.sort()
            
            memory_size_to_check = min(replay_memory_size, data_len)
            start_index = data_len % replay_memory_size
            visited = {}
            for i in range(memory_size_to_check):
                visited[i] = 0
                
            for i in range(replay_memory_size):
                pres, actions, rewards, posts, terminals, replay_indexes, heap_indexes, weights = manager.get_minibatch()
                for replay_index in replay_indexes:
                    visited[replay_index] += 1
            
            if data_len >= 100000:
                for i in range(4, memory_size_to_check / 10):
                    self.assertGreater(visited[memory_size_to_check-i-1], visited[i])
                    
    
    def test_sort(self):
        replay_memory_size = 10**6
        #data_len_list = [100, 1000, 2000, 10**6]
        data_len_list = [100, 1000, 2000]
        for data_len in data_len_list:
            minibatch_size = 32
            uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
            manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, minibatch_size, 4, self.mode, self.alpha, self.beta, self.sort_term)
            for i in range(data_len):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            #for i in range(len(manager.heap)):
            #    print 'heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])

            self.check_heap_index_list_validity(manager)
            
            startTime = time.time()
            manager.sort()
            
            print 'sort() %s data took %.1f sec' % (data_len, time.time() - startTime)

            self.check_heap_index_list_validity(manager)
                
            prev_td = sys.maxint
            for i in range(1, len(manager.heap)):
                td = manager.heap[i][1]
                self.assertGreaterEqual(prev_td, td)
                prev_td = td
            
            #for i in range(len(manager.heap)):
            #    print 'heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])
        
    def test_update_td(self):
        replay_memory_size = 1000
        data_len_list = [100, 1000, 2000, 2200]
        for data_len in data_len_list:
            minibatch_size = 32
            uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
            manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, minibatch_size, 4, self.mode, self.alpha, self.beta, self.sort_term)
            for i in range(data_len):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            self.check_heap_index_list_validity(manager)

            # Change td of the top item to the lowest then check the item is in the last
            heap_index = 1
            item = manager.heap[heap_index]
            replay_index = item[0]
            new_td = -sys.maxint
            manager.update_td(heap_index, new_td)

            item2 = manager.heap[len(manager.heap) - 1]
            replay_index2 = item2[0]
            
            self.assertEqual(replay_index, replay_index2)
            
            # Increase td of an item then check the item is in the higher index
            for i in range(100):        # Test 100 times
                heap_index = random.randint(1, len(manager.heap) - 1)
                item = manager.heap[heap_index]
                replay_index = item[0]
                td = item[1]
                new_td = td + 10
                manager.update_td(heap_index, new_td)
    
                new_heap_index = manager.heap_index_list[replay_index]
                item2 = manager.heap[new_heap_index]
                
                self.assertEquals(item[0], item2[0])
                self.assertGreaterEqual(heap_index, new_heap_index)

            self.check_heap_index_list_validity(manager)

    def test_calculate_segments(self):
        replay_memory_size = 10**6
        #data_len_list = [50000, 100000, 1000000, 2000000]
        #data_len_list = [1000000]
        data_len_list = [10000]
        start_td = 0.000001
        end_td = 1.0
        for data_len in data_len_list:
            
            print 'testing %s' % data_len
            
            td_increase = (end_td - start_td) / data_len
            minibatch_size = 32
            uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
            manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, minibatch_size, 4, self.mode, self.alpha, self.beta, self.sort_term)
            
            time1 = time.time()
            for i in range(data_len):
                state = np.zeros((84, 84), dtype=np.int)
                state.fill(i)
                #manager.add(action=0, reward=0, screen=state, terminal=0, td=start_td + i * td_increase)
                manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
    
            time2 = time.time()
            manager.calculate_segments()
            time3 = time.time()
            
            print 'Adding %d took %.1f sec.' % (data_len, time2 - time1)
            print 'Calculating segments took %.1f sec.' % (time3 - time2)

    def test_get_segments(self):
        #data_len = 10**6
        data_len = 10**4
        replay_memory_size = data_len
        minibatch_size = 32
        start_td = 0.000001
        end_td = 1.0
        td_increase = (end_td - start_td) / data_len
        uniform_replay_memory = ReplayMemory(None, False, replay_memory_size, 32, 4, 84, 84, True)            
        manager = SamplingManager(uniform_replay_memory, False, replay_memory_size, minibatch_size, 4, self.mode, self.alpha, self.beta, self.sort_term)
        for i in range(data_len):
            state = np.zeros((84, 84), dtype=np.int)
            state.fill(i)
            #manager.add(action=0, reward=0, screen=state, terminal=0, td=start_td + i * td_increase)
            manager.add(action=0, reward=0, screen=state, terminal=0, td=i)
            segment_index = manager.get_segments()
            
            if i % 100000 == 0 or i == data_len - 1:
                print 'segment_index : %s' % segment_index

        for i in range(len(manager.heap)):
            print 'manager.heap[%s] : %s, %s' % (i, manager.heap[i][0], manager.heap[i][1])
            
        manager.sort()
        segment_index = manager.get_segments()
        print 'segment_index : %s' % segment_index
        
if __name__ == '__main__':
    unittest.main()
    #TestSamplingManager().test_get_segments()
    #TestSamplingManager().test_calculate_segments()
