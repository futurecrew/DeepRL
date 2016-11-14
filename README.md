# DeepRL

This is to implement deep reinforcement learning algorithms including following papers.
  - Deep Q Network (Human-level control through deep reinforcement learning) 
  - Deep Reinforcement Learning with Double Q-learning
  - Asynchronous Methods for Deep Reinforcement Learning
  - Prioritized Experience Replay

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
