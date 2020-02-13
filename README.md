## Deep Q-Learning Lunar Lander

### **Update Notes**
As of 2/12 the project is almost complete! I estimate that there is less than one week until the final project is ready, so if you are here before then, welcome! If you are attempting to run this program on your own machine, <em>parts of it will not work</em>. Please check back again after some time to see if anything changes, and if you have any questions please contact me at tejagoparaju@gmail.com, but please include the project title in the subject of the email, or I will not respond!

**To-Do list**
There are still some things to complete before the I think the project can be completed. For the structure of the game itself:
1. An implementation of a 'reset' function that places the lander at the top of the screen when it crashes
2. An action to perform when the lander successfully lands the lander on the surface

And for the implementation of the AI:
1. Implementation of the Deep Q-Learning, in the landing_seq.py. The network needs to be linked to moon.py so that the AI can actually play the game
2. Completing and polishing up the lander_seq.py file so that everything functions and looks nice

### Acknowlegements
Special thanks to Kirill Eremenko and Hadelin de Ponteves at Super Data Science on Udemy for their course on Artificial Intelligence, and for teaching how to use the pytorch module in Python. The code in the 'lander_seq.py' file is adapted from the course they teach, and is modified by me to fit this particular project. Learn more at https://superdatascience.com

In this project I am also using John Zelle's 'graphics.py,' an open-source file (licensed under GNU General Public License) that contains a simple library for basic shapes and animation. This file is included in the repo, so no additional download is required.

### Background
In my search for something to apply what I learned I came across this demonstration of a lunar landing craft that could land itself on a simulated moon: https://gym.openai.com/envs/LunarLander-v2/ However, I noticed that there were some issues with how the simulation was constructed. First, the lander was being trained to land in between the two goalposts in the center of the screen even if the ground outside was randomly generated (which in my opinion is far more interesting to try to land on). This meant that the AI was not being encouraged to explore outside of the confines of the goalposts, so it was not truly learning to land on the 'moon,' but rather on a flat platform in the center. Another problem I saw was that the landing conditions were far too simple for the AI to have any challenge to overcome. For instance, the lander is simply falling under the influence of gravity, rather than having to correct for both horizontal and vertical velocity, which I thought was an interesting challenge for an AI lander to solve. Furthermore, since the lander always started too close to the ground, there wasn't much time before it crashed, so this also hampered the AI's ability to explore.

In attempting to fix these problems, I wanted to make a game based off the classic arcade game, Lunar Lander, and have an AI try to land on a part of the moon surface that was not constant, but instead randomly generated. Furthermore, I also wanted to give the lander more natural starting conditions, such as a velocity vector with both x- and y- components. I wanted to closely model the arcade game, while still adding my own personal spin.

#### What is Q-Learning? 
Essentially, Q-Learning is a method of reinforcement learning where an 'agent' develops a policy for taking actions in an environment based on what inputs it sees, and what rewards it receives for taking each action at its disposal. The agent attempts to maximize the reward it earns in the future, and uses what is known as a Markov Decision Process. This means that each action has a probaility of occurring, according to the Q-value the agent determined for it. Since the action it takes is not deterministic, the agent has more opportunity to explore its environment.

The 'Deep' part of Deep Q-Learning stems from the fact that there is an artificial neural network. Put simply, the network is comprised of 'neurons' that behave sort of like how our own brain's neurons function: the neuron takes an input, and gives an output, and there are several layers of neurons in the network. Each layer is connected to the adjacent layers, and there is an input layer and an output layer. In between, there are one or more 'hidden layers.' Connections between neurons have 'weights,' which just mean how strongly the neuron responds to that input. The input layer's outputs are passed to the hidden layers, and the outputs of the hidden layers are passed to the output layer. We take the q-values determined by output layer as the q-values of the agent. Since the agent still needs to learn, we back-propagate the response through the network in the reverse direction to change how the network weights the connections between its neurons. This allows the network to actually learn. For more info, check out https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning and https://en.wikipedia.org/wiki/Artificial_neural_network.

### Install instructions
In order for this project to run on your machine, first Python 3 must be installed, as well as the pytorch package. To install pytorch, visit https://pytorch.org/get-started/locally/ and follow the instructions for your configuration.


