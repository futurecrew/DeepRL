# Original code from Tambet Matiisen (https://github.com/tambetm/simple_dqn)

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, args, state_dtype, continuous_action, action_group_no):
    self.size = args.max_replay_memory
    self.minibatch_random = args.minibatch_random
    self.screen_order = args.screen_order
    self.use_color_input = args.use_color_input
    batch_size = args.train_batch_size
    history_length = args.screen_history
    height = args.screen_height
    width = args.screen_width

    # preallocate memory
    if continuous_action:
        self.actions = np.empty((self.size, action_group_no), dtype = np.float32)
    else:
        self.actions = np.empty(self.size, dtype = np.uint8)
    
    if args.use_color_input:
        self.input_channel_no = 3
    else:
        self.input_channel_no = history_length
    
    self.rewards = np.empty(self.size, dtype = np.float32)
    if self.screen_order == 'hws':        # (height, width, size)
        if args.use_color_input:
            screen_dim = (height, width, self.input_channel_no, self.size)
        else:
            screen_dim = (height, width, self.size)
        state_dim = (batch_size, height, width, self.input_channel_no)
        self.history_buffer = np.zeros((1, height, width, self.input_channel_no), dtype=state_dtype)
    else:       # (size, height, width)
        screen_dim = (self.size, height, width)
        state_dim = (batch_size, history_length, height, width)
        # Use batch_size instead of 1 to support NEON which requires batch size for network input
        self.history_buffer = np.zeros((batch_size, history_length, height, width), dtype=state_dtype)
    self.screens = np.empty(screen_dim, dtype = state_dtype)
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.history_length = history_length
    self.dims = (height, width)
    self.batch_size = batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty(state_dim, dtype = state_dtype)
    self.poststates = np.empty(state_dim, dtype = state_dtype)

    logger.info("Replay memory size: %d" % self.size)

  def add(self, action, reward, screen, terminal):
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
        if self.use_color_input:
            return self.screens[..., index]
        else:
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
    assert self.count >= self.history_length
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
        assert (index >= self.current and index - self.history_length < self.current) == False

        if self.terminals[(index - self.history_length):index].any():
            data_size_to_ret = i
            break
        
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
        if self.use_color_input:
            self.history_buffer[0, ...] = state
        else:
            if self.screen_order == 'hws':        # (height, width, size)
                self.history_buffer[0, :, :, :-1] = self.history_buffer[0, :, :, 1:]
                self.history_buffer[0, :, :, -1] = state
            else:         # (size, height, width)
                self.history_buffer[0, :-1, :, :] = self.history_buffer[0, 1:, :, :]
                self.history_buffer[0, -1, :, :] = state

  def clear_history_buffer(self):
        self.history_buffer.fill(0)
