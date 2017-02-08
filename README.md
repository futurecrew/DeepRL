# DeepRL

This project implements deep reinforcement learning algorithms including following papers.
  - Deep Q Network (Human-level control through deep reinforcement learning) 
  - Deep Reinforcement Learning with Double Q-learning
  - Asynchronous Methods for Deep Reinforcement Learning
  - Prioritized Experience Replay
  - Continuous control with deep reinforcement learning

<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c_lstm.gif" width="280">
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/breakout_a3c.gif" width="280">
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/hero.gif" width="280">



## Test scores
In my PC (i7 CPU, Titan-X Maxwell),
<p>
<nobr>
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/space_invaders_a3c.png" width="420">
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/breakout_a3c.png" width="420">
</nobr>
  - A3C FF took 20 hours for 80M global steps (nips network)
  - A3C LSTM took 44 hours for 80M global steps (nips network)

<p>
<img src="https://github.com/only4hj/DeepRL/blob/master/snapshot/hero_priority.png" width="420">
  - DQN took 96 hours for 80M steps (shown 11M steps, nature network)
  - Double-Q took 112 hours for 80M steps (shown 11M steps, nature network)
  - Prioritized took 112 hours for 80M steps (shown 11M steps, nature network)


## Torcs

<a href="https://www.youtube.com/watch?v=CtuDq1SmwJM&feature=youtu.be" target="_blank"><img src="https://raw.githubusercontent.com/only4hj/DeepRL/master/snapshot/torcs-1.png" width="560" height="315" border="10" /></a>

After training in simulator Torcs, it learns how to accelerate, brake and turn the steering wheel.
<br>
Click the image to watch the video.
<br>

## Requirements
  - Python-2.7
  - pip, scipy, matplotlib, numpy
  - Tensorflow-0.11
  - Arcade-Learning-Environment
  - Torcs (optional)
  - Vizdoom (in working)
  <br><br>
  See <a href="INSTALL.md">this</a> for installation.
  
## How to train
```
DQN         : python train.py /path/to/rom --drl dqn
Double DQN  : python train.py /path/to/rom --drl double_dqn
Prioritized : python train.py /path/to/rom --drl prioritized_rank
A3C FF      : python train.py /path/to/rom --drl a3c --thread-no 8
A3C LSTM    : python train.py /path/to/rom --drl a3c_lstm --thread-no 8
DDPG        : python train.py torcs --ddpg
```
  
## How to retrain
```
python train.py /path/to/rom --drl a3c --thread-no 8 --snapshot path/to/snapshot_file
ex) python train.py /rom/breakout.bin --drl a3c --thread-no 8 --snapshot snapshot/breakout/20161114_003838/a3c_6250000
```

## How to play
```
python play.py path/to/snapshot_file
ex) python play.py snapshot/space_invaders/20161114_003838/a3c_79993828
```

## Debug console commands
While training you can send several debug commands in the console.
- p : print debug logs or not
- u : pause training or not
- quit : finish running
- d : show the current running screen or not. You can see how the training is going on in the game screen.
- - : show the screen more fast
- + : show the screen more slowly


## Reference projects
  - https://github.com/tambetm/simple_dqn
  - https://github.com/miyosuda/async_deep_reinforce
  - https://github.com/muupan/async-rl
  - https://github.com/yanpanlau/DDPG-Keras-Torcs
