**Environment description**

1.What is the agent?

-> A DQN agent is a value-based reinforcement learning agent that takes current state, picks the best action based on model prediction and executes it for which the environment returns the reward and the next state.

In the Implementation, class DQNAgent is an agent.

2.What are the possible actions?

->There are two possible actions, an agent can movie cart either   left or right i.e 1 or 0.

3.What are the observations?

->There are four observations:


*   Cart Position from -4.8 to 4.8
*   Cart Velocity from -Inf to Inf
*   Pole Angle from -24° to 24°
*   Pole Velocity At Tip from -Inf to Inf

4.What are the rewards?

->In each timestep, a reward of +1 is given for the pole remains   upright and a reward of -1 incase the cartpole is terminated.

**Hyperparameters**

5.What are the hyper parameters for DQN?  How did you choose the hyper parameters?

->Following are the hyperparameters for a DQN model:
* Epiosdes - number of iteration the agent is into play
* gamma - decay or discount rate, to calculate the future discounted reward.
* epsilon - This is the rate in which an agent randomly decides its action rather than prediction. It is also called exploration rate.
* epsilon_decay - decrease rate of the number of explorations as it gets good at playing games.
* epsilon_min - It is the minimum exploration rate at which we want the agent to explore..
* learning_rate - Determines how much neural net learns in each iteration and is used to smooth the updates 
* batch_size - number of inputs to the DQN model.

Among these various tunable parameters, we have seleceted only two of them,gamma and learning_rate, to tune in our code. We have taken gamma = [0.95, 0.85, 0.75] and learning_rate = [0.01, 0.001, 0.0001] with the total of nine combinations of the parameters.The rest of the parameters are set as follows:
* Episodes = 300
* epsilon = 1
* epsilon_decay = 0.995
* epsilon_min = 0.01
* batch_size = 32
* gamma = [0.95, 0.85, 0.75]
* learning_rate = [0.01, 0.001, 0.0001]

For each combination of parameters, we trained the cartpole system for 300 episodes and plotted the the reward function. From the plot above in the program execution , we can say that the best parameter is gamma = 0.95 and learning rate = 0.001 as the target of 500 score is reached earlier and multiple times during 300 episodes of trianing.

Similarly, all the hyperparameters could be tested over the range of values in the similar way but it takes considerably large time, so we took only two parameters for tuning in this assignment.

**Reward plot**

The figure below is the plot of accumulated rewards versus episodes with gamma = 0.95 and learning rate = 0.001. We can see that the first target score of 500 is reached before 100 episodes. Similarly, we can observe that the agent has reached to the target score multiple times in the plot.

![alt text](https://drive.google.com/uc?id=1hBSpRvCbZTDDRruZJggyfXKG-S61zQEF)


**Implementation**

1.Neural Network

We have used keras to create a neural network model with 2 hidden layers having 24 neurons on each layer. Input layer has four neuron equal to the number of observation space. The output layer has two neuron equal equal to the number of action space. we have used model.fit (next_state,reward) available in Keras NN model to train the network. The gap between prediction and target value is decreased by the learning rate using this fit function.We have used "mse" as the loss function. As we go on updating this process, the loss will decrease and score will grow higher. After training, the model will predict the reward of current state based on the data we trained.

Layer | Nodes | Activation
---|---|:---|
Input| 4 | 
Dense|24|ReLU
Dense|24|ReLU
Output|2|Linear

2.Remember function

The DQN has property to forget the previus expereinces and overwrites them with new experiences. So, we need to create memory of previous experiences and observations to re-train the model. In this implementation, remember function is used to store state, action, reward and next state to the memory (list)

3.train_agent function

This function trains the NN with some randomly sampled experiences from the memory. We update the Q value with the cumulative discounted future rewards using the gamma value that takes into consideration both the immediate rewards and future rewards and makes the agent perfofrm well in long-term.

4.Act function

At first, the agent doesn't have knowledge of patterns of action so it randomly selects its action by a certain percentage, called epsilon. Later, the agent predicts the reward based on the current state and the action corresponding to the highest reward is picked.

5.Run Function

The goal is to balance the pole upright by moving the cart either left or right and the environment is successful if the pole is balanced for 500 frames. So every frame with the pole balanced gets +1 reward and our target is to accumulate 500 rewards. In our code, we are running for 300 episodes to train the agent. In each episodes, we reset the environment and decide the action usnig "act function" and advance the game to the next frame based on the action. We store results of each step to memory using "remember function", which we use for training on every step using "train_agent function".



