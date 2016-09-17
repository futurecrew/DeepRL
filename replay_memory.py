# Original code from Tambet Matiisen (https://github.com/tambetm/simple_dqn)

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, be, use_gpu_replay_mem, size, batch_size, history_no, width, height, minibatch_random):
    self.be = be
    self.use_gpu_replay_mem = use_gpu_replay_mem
    self.size = size
    self.minibatch_random = minibatch_random
    # preallocate memory
    self.actions = np.empty(self.size, dtype = np.uint8)
    self.rewards = np.empty(self.size, dtype = np.integer)
    if self.use_gpu_replay_mem:
        self.screens = self.be.empty((self.size, height, width), dtype = np.uint8)
    else:
        self.screens = np.empty((self.size, height, width), dtype = np.uint8)
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.history_length = history_no
    self.dims = (height, width)
    self.batch_size = batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    if self.use_gpu_replay_mem:
        self.prestates = self.be.empty((self.batch_size, self.history_length,) + self.dims, dtype=np.uint8)
        self.poststates = self.be.empty((self.batch_size, self.history_length,) + self.dims, dtype=np.uint8)
        self.prestates_view = [self.prestates[i, ...] for i in xrange(self.batch_size)]
        self.poststates_view = [self.poststates[i, ...] for i in xrange(self.batch_size)]
    else:
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)

    logger.info("Replay memory size: %d" % self.size)

  def add(self, action, reward, screen, terminal):
    assert screen.shape == self.dims
    addedIndex = self.current
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
    #logger.debug("Memory count %d" % self.count)
    
    return addedIndex
  
  def get_state(self, index):
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def get_minibatch(self):
    if self.minibatch_random:
        return self.get_minibatch_random()
    else:
        return self.get_minibatch_sequential()
    
  def get_minibatch_random(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      if self.use_gpu_replay_mem:      
          self.prestates_view[len(indexes)][:] = self.get_state(index - 1)
          self.poststates_view[len(indexes)][:] = self.get_state(index)
      else:            
          self.prestates[len(indexes), ...] = self.get_state(index - 1)
          self.poststates[len(indexes), ...] = self.get_state(index)
      indexes.append(index)

    # copy actions, rewards and terminals with direct slicing
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return self.prestates, actions, rewards, self.poststates, terminals

  def get_minibatch_sequential(self):
    # memory must include poststate, prestate and history
    assert self.count >= self.batch_size + self.history_length
    # sample random indexes
    indexes = []
    for i in range(self.count):
        index = self.current - i - 1
        if index < 0:
            index += self.count
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        
        # NB! having index first is fastest in C-order matrices
        if self.use_gpu_replay_mem:      
            self.prestates_view[len(indexes)][:] = self.get_state(index - 1)
            self.poststates_view[len(indexes)][:] = self.get_state(index)
        else:            
            self.prestates[len(indexes), ...] = self.get_state(index - 1)
            self.poststates[len(indexes), ...] = self.get_state(index)
        indexes.append(index)

        if len(indexes) == self.batch_size:
            break

    # copy actions, rewards and terminals with direct slicing
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return self.prestates, actions, rewards, self.poststates, terminals
