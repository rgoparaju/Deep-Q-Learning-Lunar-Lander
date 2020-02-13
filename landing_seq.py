# AI for Deep Q-Learning Lunar Lander

# Import the required libraries. The torch library and its extensions are necessary for building
# the neural network and implementing deep q-learning. The os library is used for saving the
# 'brain' of the AI, so that it can be reused even after the program is closed

import os

# pytorch libraries. torch.nn contains the tools to implement the neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optimizer to perform Stochastic Gradient Descent, which allows the network to back-propagte what
# it has learned to update how it determines what action to take based on the rewards it receives
import torch.optim as optim 

import torch.autograd as autograd
# This class is used to convert tensors into a variable that contains a gradient
from torch.autograd import Variable 

# Neural Network class - there are three layers of 'neurons', the input layer, one hidden layer,
# and finally the output layer, that gives q-values that determine what action the lander takes.
class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, actions):
        super(NeuralNetwork, self).__init__() # Trick to use methods in Module
        
        # input_size is the size of the vector that encodes the input state for the lander at all
        # times. There are 5 inputs: the rotation angle of the lander, the x-speed, the y-speed, 
        # the slope of the ground beneath the lander, and the height of the lander from the ground.
        # This also corresponds to the number of input neurons in the network
        self.input_size = input_size
        
        # Final output action the network determines. There are three actions: rotate left, rotate
        # right, or fire the thruster. This also corresponds to the number of neurons in the output
        # layer. 
        self.actions = actions
        
        # A full connection is the set of connections between each layer of neurons. Since there are
        # two hidden layers, there are 3 full connections between each layer. 30 means that there
        # are 30 neurons in each layer.
        self.full_connection_1 = nn.Linear(input_size, 30)
        self.full_connection_2 = nn.Linear(30,30)
        self.full_connection_3 = nn.Linear(30, actions)
    
    # Function that activates the neurons, using the rectifier function which is used because this
    # is a nonlinear problem.
    def forward(self, state): # state is the input to the neural network
        x = F.relu(self.full_connection_1(state)) # relu - rectifier
        y = F.relu(self.full_connection_2(x))
        q_values = self.full_connection_3(y)
        
        # The q-values are essentially how the AI determines which action to take. If a given action
        # has a higher q-value, it means that the AI has a higher probability of choosing that 
        # action. Since we are implementing a Markov Decision Process, the AI is not guaranteed to
        # choose the action with the highest q-value all the time, in order to encourage exploration
        # of the environment. 
        return q_values

# Experience Replay - allows the AI to learn from the previous rewards it has received to determine how 
# it should proceed with learning. 

class ExperienceReplay(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

""" !!! Still a work in progress !!! Check back again at another time!
class Deep_Q_Network():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = NeuralNetwork(input_size, nb_action)
        self.memory = ExperienceReplay(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
"""    
