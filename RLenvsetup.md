# Setiing up the environment
The commands valid as of 04-10-2020
This additional package assumes you have other standard dependencies installed. If not and it causes error in the middle, please follow the message to troubleshoot or reach out to us for help.
This is for a local machine in a linux server, and has not been tested in the other system. If you use a VM service such as google colab and encounter a problem, please reach out to us.
Unless your project involves computer vision tasks and neural network models, you may not need a gpu machine, so you can still try in your local machine even if there is no (nvidia or compatible) gpu. 
```
pip install -U pip
pip install -U torch torchvision ## caution: this is for linux. Windows' pip won't work and need conda: see https://pytorch.org/get-started/locally/
pip install gym
pip install atari-py
pip install opencv-python
pip install -U scipy
pip install -U tensorflow-gpu ## you won't need it if you don't have a gpu or in a colab environment where tensorflow-gpu is already available.
pip install -U tensorboard
