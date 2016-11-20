# DeepRL

This is to implement deep reinforcement learning algorithms including following papers.
  - Deep Q Network (Human-level control through deep reinforcement learning) 
  - Deep Reinforcement Learning with Double Q-learning
  - Asynchronous Methods for Deep Reinforcement Learning
  - Prioritized Experience Replay

<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c_lstm.gif" width="300">


## Test scores
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c.png" width="400">

## Requirements
  - Python2.7
  - Arcade-Learning-Environment
  - Tensorflow 1.0
  - cv2
  - Vizdoom (optional)
  - Neon (optional)
  
## How to train
```
python deep_rl_train.py [path_to_rom_file] --multi-thread-no 8
```

## How to run
```
python play.py [path_to_rom_file] --replay-file [path_to_snapshot_file]
e.g. python play.py space_invaders.bin --replay-file snapshot/space_invaders/20161114_003838/a3c_40425623
```

## Reference projects
  - https://github.com/tambetm/simple_dqn
  - https://github.com/miyosuda/async_deep_reinforce
  - https://github.com/muupan/async-rl
