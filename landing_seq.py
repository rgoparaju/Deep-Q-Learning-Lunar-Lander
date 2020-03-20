# AI for Deep Q-Learning Lunar Lander

# Import the required libraries. The torch library and its extensions are necessary for building
# the neural network and implementing deep q-learning. The os library is used for saving the
# 'brain' of the AI, so that it can be reused even after the program is closed

import os
import random
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

# Neural Network class - there are three layers of 'neurons', the input layer, two hidden layers,
# and finally the output layer, that gives q-values that determine what action the lander takes.
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, actions):
        super(NeuralNetwork, self).__init__() # Trick to use methods in Module
        
        # input_size is the size of the vector of the input state for the lander at all times.
        # There are 5 inputs: the rotation angle of the lander, the x-speed, the y-speed, 
        # the slope of the ground beneath the lander, and the height of the lander from the ground.
        self.input_size = input_size
        
        # Final output action the network determines. There are three actions: rotate left, rotate
        # right, or fire the thruster. This also corresponds to the number of neurons in the output
        # layer. 
        self.actions = actions
        
        # A full connection is the set of connections between each layer of neurons. Since there are
        # two hidden layers, there are 3 full connections between each layer. 30 means that there
        # are 30 neurons in each layer.
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30, actions)
    
    # Rectifier function activates the neurons, used because this is a nonlinear problem.
    def forward(self, state): # state is the input to the neural network
        x = F.relu(self.fc1(state))
        y = F.relu(self.fc2(x))
        q_values = self.fc3(y)
        
        # The q-values are how the AI determines which action to take. If a given action has a 
        # higher q-value, it means that the AI has a higher probability of choosing that action.
        # Since we are implementing a Markov Decision Process, the AI is not guaranteed to choose
        # the action with the highest q-value all the time, in order to encourage exploration of 
        # the environment. 
        return q_values

# Experience Replay
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

class Landing_Sequence():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = NeuralNetwork(input_size, nb_action)
        self.memory = ExperienceReplay(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*50) # T=50
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_landing_seq.pth')
    
    def load(self):
        if os.path.isfile('last_landing_seq.pth'):
            print("Loading...")
            checkpoint = torch.load('last_landing_seq.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded")
        else:
            print("No Sequence Found")
