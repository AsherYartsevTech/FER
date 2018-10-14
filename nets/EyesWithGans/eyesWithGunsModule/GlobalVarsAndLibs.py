# imports for every module
import random
import tensorflow as tf
from tensorflow.contrib.layers import *
from tensorflow import initializers
import numpy as np


'''GLOBAL VARIABLES'''

training_iters = 200 
epochsNum = 30
learning_rate = 0.001 
batch_size = 512
image_size = 48
n_input = 48
n_classes=7
num_channels = 1
input_n_channels = 1

'''weighted average on net classification parameters'''
brain_weight = 0.5
eye1_weight = 0.3
eye2_weight = 0.2


