# 20BN-jester
Deep Learning project written with Keras/TensorFlow to predict human hand gestures based on the jester dataset

## Getting started

### Library version
The version of the libraries used are:
Keras: 2.2.2
TensorFlow: 1.10.0

### Usage
```
python main.py --config <configuration_file>
```
Example:
```
python main.py --config "config.cfg"
```

## Comments
My current best result is currently 85.99% top-1 accuracy on the [20 BN jester leaderboard](https://20bn.com/datasets/jester/).
It was obtained by training a 3D version of ResNet101 with a dropout of 0.5 between each block. 
