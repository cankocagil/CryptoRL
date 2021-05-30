
import os, copy, time, sys, random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd, numpy as np
from collections import deque
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam, RMSprop
#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')

from models.PPO import Actor_Model, Critic_Model, Shared_Model
from models.PPO_cont import SharedModelContinious
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing


class Agent:
    """ A custom Bitcoin trading agent """
    def __init__(self, lookback_window_size=50, lr=0.00005,
                 epochs=1, optimizer=Adam, batch_size=32,
                 num_indicators = 9, model="MLP"):
                
        self.lookback_window_size = lookback_window_size
        self.model = model
        
        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        self.pretrained_path = 'pretrained'
        # folder to save models
        self.log_name = os.path.join(self.pretrained_path, datetime.now().strftime("%Y-%m-%d %H: %M")+ "_trader")

       
        # State size contains Market+Orders history for the last lookback_window_size steps
        #self.state_size = (lookback_window_size, 10)
        self.state_size = (lookback_window_size, 10 + num_indicators) # 10 standard information +9 indicators

        # Neural Networks part bellow
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create shared Actor-Critic network model
        self.Actor = self.Critic = Shared_Model(input_shape=self.state_size, action_space = self.action_space.shape[0],
                                                lr=self.lr, optimizer = self.optimizer, model=self.model)
        # Create Actor-Critic network model
        #self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        #self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        
    # create tensorboard writer
    def create_writer(self,
                     initial_balance, 
                     normalize_value,
                     train_episodes,
                     reward_strategy:str = 'base'):

        self.replay_count = 0
        self.writer = SummaryWriter('runs/'+self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.start_training_log(initial_balance,
                                normalize_value,
                                train_episodes,
                                reward_strategy)
            
    def start_training_log(self, initial_balance, normalize,
                           train_episodes, reward_strategy:str = 'incremental'):   

        # save training parameters to Parameters.txt file for future
        with open(self.log_name+"/Parameters.txt", "w") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training start: {current_date}\n")
            params.write(f"initial_balance: {initial_balance}\n")
            params.write(f"training episodes: {train_episodes}\n")
            params.write(f"lookback_window_size: {self.lookback_window_size}\n")
            params.write(f"lr: {self.lr}\n")
            params.write(f"epochs: {self.epochs}\n")
            params.write(f"batch size: {self.batch_size}\n")
            params.write(f"normalize : {normalize}\n")
            params.write(f"model: {self.model}\n")
            params.write(f"reward strategy: {reward_strategy}")
            
    def end_training_log(self):
        with open(self.log_name + "/Parameters.txt", "a+") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training end: {current_date}\n")

    def normalize_reward(self, reward, eps = 1e-6):
        return (reward - reward.mean()) / (reward.std() + eps)


    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

            
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)
        
        # Compute advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state):

        """ Use the network to predict the next action to take, using the model """
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction
        
    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        self.Actor.Actor.save_weights(f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(f"{self.log_name}/{score}_{name}_Critic.h5")

        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                arguments = ""
                for arg in args:
                    arguments += f", {arg}"
                log.write(f"{current_time}{arguments}\n")

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(os.path.join(folder, f"{name}_Critic.h5"))




class AgentContinious(Agent):
    """ A custom Bitcoin trading agent """
    def __init__(self, 
                 lookback_window_size=50,
                 lr=0.00005,
                 epochs=1,
                 optimizer=Adam,
                 batch_size=32,
                 model=""):

        super(AgentContinious, self).__init__(
            lookback_window_size,
            lr,
            epochs,
            optimizer,
            batch_size,
            model
        )
                 
       
        # Create shared Actor-Critic network model
        self.Actor = self.Critic = SharedModelContinious(input_shape=self.state_size, 
                                                         action_space = self.action_space.shape[0],
                                                         lr=self.lr,
                                                         optimizer = self.optimizer,
                                                         model=self.model)
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        props = tanh2prob(prediction)
        action = np.random.choice(self.action_space, p=props)
        return action, prediction


def tanh2prob(prediction):
    onehot = np.zeros(3)
    action = discretize_action(prediction)
    onehot[action] = 1
    return onehot


def discretize_action(action):   
    if 0.1 > action > - 0.1:
        action = 0 # Hold
    elif action >= 0.1 :
        action = 1 # Buy
    else:
        action = 2 # Sell
    return action