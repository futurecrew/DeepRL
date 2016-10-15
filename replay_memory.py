# Original code from Tambet Matiisen (https://github.com/tambetm/simple_dqn)

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, size, batch_size, history_length, width, height, minibatch_random, screen_order='shw'):
    self.size = size
    self.minibatch_random = minibatch_random
    self.screen_order = screen_order
    # preallocate memory
    self.actions = np.empty(self.size, dtype = np.uint8)
    self.rewards = np.empty(self.size, dtype = np.integer)
    if self.screen_order == 'hws':        # (height, width, size)
        screen_dim = (height, width, self.size)
        state_dim = (batch_size, height, width, history_length)
        self.history_buffer = np.zeros((1, height, width, history_length), dtype=np.float32)
    else:       # (size, height, width)
        screen_dim = (self.size, height, width)
        state_dim = (batch_size, history_length, height, width)
        self.history_buffer = np.zeros((1, history_length, height, width), dtype=np.float32)
    self.screens = np.empty(screen_dim, dtype = np.uint8)
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.history_length = history_length
    self.dims = (height, width)
    self.batch_size = batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty(state_dim, dtype = np.uint8)
    self.poststates = np.empty(state_dim, dtype = np.uint8)

    logger.info("Replay memory size: %d" % self.size)

  def add(self, action, reward, screen, terminal):
    assert screen.shape == self.dims
    addedIndex = self.current
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    if self.screen_order == 'hws':        # (height, width, size)
        self.screens[..., self.current] = screen
    else:           # (size, height, width)
        self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
    #logger.debug("Memory count %d" % self.count)
    
    self.add_to_history_buffer(screen)
    
    return addedIndex
  
  def get_state(self, index):
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
        # use faster slicing
        if self.screen_order == 'hws':        # (height, width, size)
          return self.screens[..., (index - (self.history_length - 1)):(index + 1)]
        else:           # (size, height, width)
          return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      if self.screen_order == 'hws':        # (height, width, size)
        return self.screens[..., indexes]
      else:           # (size, height, width)
        return self.screens[indexes, ...]

  def get_current_state(self):
    if self.current == 0:
        index = self.size - 1
    else:
        index = self.current - 1
    return self.get_state(index)
  
  def get_minibatch(self, max_size=-1):
    if self.minibatch_random:
        return self.get_minibatch_random(max_size)
    else:
        return self.get_minibatch_sequential(max_size)
    
  def get_minibatch_random(self, max_size):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    if max_size == -1:
        max_size = self.batch_size
        
    # sample random indexes
    indexes = []
    while len(indexes) < min(self.batch_size, max_size):
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
      self.prestates[len(indexes), ...] = self.get_state(index - 1)
      self.poststates[len(indexes), ...] = self.get_state(index)
      indexes.append(index)

    # copy actions, rewards and terminals with direct slicing
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    return self.prestates, actions, rewards, self.poststates, terminals

  def get_minibatch_sequential(self, max_size):
    # memory must include poststate, prestate and history
    assert self.count >= self.batch_size + self.history_length
    if max_size == -1:
        max_size = self.batch_size
        
    data_size_to_ret = min(self.batch_size, max_size)

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
        self.prestates[len(indexes), ...] = self.get_state(index - 1)
        self.poststates[len(indexes), ...] = self.get_state(index)
        indexes.append(index)

        if len(indexes) == data_size_to_ret:
            break

    # copy actions, rewards and terminals with direct slicing
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    
    return self.prestates[:data_size_to_ret, ...], actions, rewards, self.poststates[:data_size_to_ret, ...], terminals

  def add_to_history_buffer(self, state):
        self.history_buffer[0, :, :, :-1] = self.history_buffer[0, :, :, 1:]
        self.history_buffer[0, :, :, -1] = state

  def clear_history_buffer(self):
        self.history_buffer.fill(0)
