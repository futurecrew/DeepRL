## This guide shows how to install torcs environment in Ubuntu.

- This project is tested in python-2.7 and Ubuntu 14.04.
  <br>
  To run Torcs the following packages are required.

- xautomation (http://linux.die.net/man/7/xautomation)
  <br>
  sudo apt-get update
  sudo apt-get install xautomation
- Gym-TORCS
  <br>
  sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev 
  Download source from https://github.com/ugo-nama-kun/gym_torcs
  Untar the archive
  cd into the vtorcs-RL-color directory
  ./configure
  make
  make install
  make datainstall

- Initialization of the Race
  Go to the link below and follow instructions.
  https://github.com/ugo-nama-kun/gym_torcs#initialization-of-the-race
  
## How to train
```
Without vision   : python train.py torcs --ddpg
With vision only : python train.py torcs --ddpg --vision
```
