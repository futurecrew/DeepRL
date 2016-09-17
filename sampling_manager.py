import random
from replay_memory import ReplayMemory
import numpy as np

class SamplingManager:
    def __init__(self, replay_memory, use_gpu_replay_mem, size, batch_size, history_len, sampling_mode,
                            sampling_alpha, sampling_beta, sort_term):
        self.replay_memory = replay_memory
        self.use_gpu_replay_mem = use_gpu_replay_mem
        self.batch_size = batch_size
        self.history_len = history_len
        self.sampling_mode = sampling_mode
        self.alpha = sampling_alpha
        self.beta = sampling_beta
        self.sort_term = sort_term
        self.heap_index_list = [-1] * size        # This list maps replay_index to heap_index
        self.heap = []                                     # Binary heap
        self.heap.append((None, None))
        self.proportion_epsilon = 0.0000001
        self.max_weight= 0
        self.max_td = 1.0
        self.segment_calculation_unit = 1000
        self.segment_index = {}              # heap indexes for each segment
        self.add_call_no = 0

    @property
    def count(self):
        return self.replay_memory.count

    def add(self, action, reward, screen, terminal, td=None):
        if td == None:
            td = self.max_td
        added_replay_index = self.replay_memory.add(action, reward, screen, terminal)

        # If there was the same data then remove it first
        heap_index = self.heap_index_list[added_replay_index]
        if heap_index != -1:
            self.remove(heap_index)
        
        item = (added_replay_index, td)
        self.heap.append(item)
        child_index = len(self.heap) - 1
        self.heap_index_list[added_replay_index] = child_index
        self.reorder_upward(child_index)
        
        self.add_call_no += 1
        
        if self.add_call_no % self.sort_term == 0:
            self.sort()
        if self.add_call_no % (10**5) == 0:     # Clear segment_index to calculate segment again
            self.segment_index = {}
        
    def remove(self, index):
        last_index = len(self.heap) - 1
        self.heap_index_list[self.heap[index][0]] = -1
        if index == last_index:
            self.heap.pop(last_index)
        else:
            self.heap[index] = self.heap[last_index]
            self.heap_index_list[self.heap[index][0]] = index
            self.heap.pop(last_index)
            self.reorder(index)
        
    def get_top(self):
        return self.heap[1]
    
    def get(self, index):
        return self.heap[index]
    
    def get_heap_length(self):
        return len(self.heap) - 1
    
    def swap(self, index1, item1, index2, item2):
        self.heap[index1] = item1
        self.heap[index2] = item2
        self.heap_index_list[item1[0]] = index1
        self.heap_index_list[item2[0]] = index2

    def reorder(self, index, newValue=None):
        if newValue != None:
            self.heap[index] = (self.heap[index][0], newValue)

        reordered = self.reorder_upward(index)
        
        if reordered == False:
            self.reorder_downward(index) 
        
    def reorder_upward(self, index):
        reordered = False
        child_index = index

        while True:
            item = self.heap[child_index]
            parent_index = child_index / 2
            parent_item = self.heap[parent_index]
            if parent_index == 0 or item[1] <= parent_item[1]:
                break
            self.swap(parent_index, item, child_index, parent_item)            
            child_index = parent_index
            reordered = True
            
        return reordered
        
    def reorder_downward(self, index):
        parent_index = index
        while True:
            parent_item = self.heap[parent_index]
            child_index1 = parent_index * 2
            child_index2 = parent_index * 2 + 1
            
            if child_index2 > len(self.heap) - 1:
                if child_index1 <= len(self.heap) - 1 and self.heap[child_index1][1] > parent_item[1]:
                    self.swap(parent_index, self.heap[child_index1], child_index1, parent_item)
                    self.heap_index_list[self.heap[child_index1][0]] = parent_index
                    self.heap_index_list[parent_item[0]] = child_index1
                    
                    parent_index = child_index1
                else:
                    break
            else:
                if self.heap[child_index1][1] > parent_item[1] and self.heap[child_index1][1] >= self.heap[child_index2][1]:
                    self.swap(parent_index, self.heap[child_index1], child_index1, parent_item)
                    parent_index = child_index1
                elif self.heap[child_index2][1] > parent_item[1] and self.heap[child_index2][1] >= self.heap[child_index1][1]:
                    self.swap(parent_index, self.heap[child_index2], child_index2, parent_item)
                    parent_index = child_index2
                else:
                    break
            
    def reorder_top(self, new_top_value):
        top = self.get_top()
        self.heap[1] = (top[0], new_top_value) 
        
        self.reorder_downward(1) 

    def sort(self):
        new_heap = []
        new_heap.append((None, None))
        heap_size = len(self.heap)
        #print 'heap_size : %s' % heap_size
        for i in range(1, heap_size):
            #print 'i : %s' % i
            top = self.get_top()
            new_heap.append(top)
            last_index = heap_size - i
            #print 'last_index : %s' % last_index
            self.heap[1] = self.heap[last_index]
            self.heap.pop(last_index)
            if last_index > 1:
                self.reorder_downward(1)
            
        self.heap = new_heap        
        
        for i in range(1, heap_size):
            self.heap_index_list[self.heap[i][0]] = i
            
        self.segment_index = {}
        
    def update_td(self, heap_index, td):
        if td > self.max_td:
            self.max_td = td
        self.reorder(heap_index, td)
    
    def get_segments(self):
        data_len = len(self.heap) - 1
        segment = data_len / self.segment_calculation_unit * self.segment_calculation_unit
        if segment == 0:       # If data len is less than necessary size then use uniform segments
            return None
        else:
            if segment not in self.segment_index:
                self.segment_index[segment] = self.calculate_segments(segment)
            return self.segment_index[segment]

    def get_p(self, heap_index):
        if self.sampling_mode == 'RANK':
            return (1.0 / heap_index) ** self.alpha
        elif self.sampling_mode == 'PROPORTION':
            return (abs(self.heap[heap_index][1]) + self.proportion_epsilon) ** self.alpha
        
    def calculate_segments(self, data_len=None):
        if data_len == None:
            data_len = len(self.heap)
            
        self.total_psum = 0
        for i in range(1, data_len):
            self.total_psum += self.get_p(i)

        segment = self.total_psum / self.batch_size
        segment_sum = 0
        segment_no = 1
        segment_index = []
        for i in range(1, data_len):
            segment_sum += self.get_p(i)
            if segment_sum >= segment * segment_no:
                segment_index.append(i)
                segment_no += 1
                if len(segment_index) == self.batch_size - 1:
                    segment_index.append(len(self.heap) - 1)
                    break
        
        """
        for i in range(len(self.heap)):
            print 'self.heap[%s] : %s, %s' % (i, self.heap[i][0], self.heap[i][1])
        print 'segment_index : %s' % segment_index
        """
        return segment_index
            
    def get_minibatch(self):
        segment_index = self.get_segments()
        
        # sample random indexes
        indexes = []
        heap_indexes = []
        weights = []        
        for segment in range(self.batch_size):
            if segment_index == None:
                    index1 = 1
                    index2 = self.count
            else:
                if segment == 0:
                    index1 = 1
                else:
                    index1 = segment_index[segment-1] + 1
                index2 = segment_index[segment]

            # find index 
            while True:
                heap_index = random.randint(index1, index2)
                replay_index = self.heap[heap_index][0]
                
                repeat_again = False
                if replay_index < self.history_len:
                    repeat_again = True
                # if wraps over current pointer, then get new one
                if replay_index >= self.replay_memory.current and replay_index - self.history_len < self.replay_memory.current:
                    repeat_again = True
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.replay_memory.terminals[(replay_index - self.history_len):replay_index].any():
                    repeat_again = True
                
                if repeat_again:
                    self.reorder(heap_index, 0)         # Discard and never use this data again
                    continue            

                if segment_index == None:
                    weight = 1.0
                else:
                    weight = (self.total_psum / self.get_p(heap_index) / len(self.heap)) ** self.beta
                    if weight > self.max_weight:
                        self.max_weight = weight
                    weight = weight / self.max_weight
                weights.append(weight)
                break
                
            # NB! having index first is fastest in C-order matrices
            if self.use_gpu_replay_mem:
                self.replay_memory.prestates_view[len(indexes)][:] = self.replay_memory.get_state(replay_index - 1)
                self.replay_memory.poststates_view[len(indexes)][:] = self.replay_memory.get_state(replay_index)
            else:            
                self.replay_memory.prestates[len(indexes), ...] = self.replay_memory.get_state(replay_index - 1)
                self.replay_memory.poststates[len(indexes), ...] = self.replay_memory.get_state(replay_index)
            indexes.append(replay_index)
            heap_indexes.append(heap_index)
    
        # copy actions, rewards and terminals with direct slicing
        actions = self.replay_memory.actions[indexes]
        rewards = self.replay_memory.rewards[indexes]
        terminals = self.replay_memory.terminals[indexes]
        return self.replay_memory.prestates, actions, rewards, self.replay_memory.poststates, terminals, indexes, heap_indexes, weights

        