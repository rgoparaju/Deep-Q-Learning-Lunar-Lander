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
    
    # Rectifier function activates the neurons during forward propagation. The rectifier is used
    # because this is a nonlinear problem.
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

# Experience Replay is used to store the past states and rewards the neural network earned so that
# it can sample them to learn how to make its next decision. It can be considered a "long-term"
# memory for the network. 
class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    # This method adds new experiences to the class's memory, as well as making sure the memory
    # does not exceed its capacity. The event object is a tuple which contains the last state of
    # the lander, the new upcoming state, the last action performed, and the last reward earned.
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    # Returns a random sample of the elements in the memory object. The zip(*) function groups
    # the states, actions and rewards that are in each memory tuple, to be mapped into pytorch 
    # Variables that have associated gradients. This is used later when performing stochastic
    # gradient descent. 
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# This class takes an object of the NeuralNetwork class created above and performs deep q-learning
# with it. This is like a 'pilot' that is tasked with learning how to land the lunar lander. 
class Landing_Sequence():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        
        # Sliding window of the mean of the last 100 rewards the AI received
        self.reward_window = [] 
        
        self.model = NeuralNetwork(input_size, nb_action)
        self.memory = ExperienceReplay(100000)
        
        # The optimizer is used to help perform stochastic gradient descent during back-propagation.
        # lr is called the learning rate, which helps the AI learn to explore its environment.
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        
        # Though the input comes in as a list, it needs to be changed into a Torch tensor with an 
        # extra dimension. This extra dimension is added by using the 'unsqueeze' trick. This is
        # because the network expects the input state to come in as a batch.
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        self.last_action = 0
        self.last_reward = 0
    
    # This function takes the input state vector as a parameter and using the softmax function,
    # generates a probability for each q-value. These q-values correspond to the actions the 
    # lander can take; that is, rotate clockwise, counterclockwise, or fire the thruster. The
    # most favorable action will have the highest probability of occurring.
    # This input still needs to be converted to a Variable object, but since we do not require
    # an associated gradient, we set requires_grad to False.
    # The T parameter will be able to change how 'certain' the network is of taking any given
    # action, so changing it will change the learning behavior (higher is more certain).
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, requires_grad = True))*50) # T=50
        action = probs.multinomial(1)
        return action.data[0,0] # Trick to separate the value from the extra dimension in the batch.
    
    # Function that takes care of forward propagation, calculating the losses and back propagation.
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # The gather function is a trick used to only extract the action the network selects as 
        # its output. However since batch_action does not have an added dimension like batch_state,
        # we need to use the unsqueeze function. Finally, squeeze collapses this extra batch
        # dimension to turn back into a simple vector of outputs.
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
