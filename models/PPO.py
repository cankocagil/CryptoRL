
from pickletools import optimize
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras import backend as K
#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.optimizers import Adam, RMSprop


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


class Shared_Model:
    def __init__(self, input_shape, action_space, lr, optimizer, model="Dense"):
        """[summary]

        Args:
            input_shape ([type]): [description]
            action_space ([type]): [description]
            lr ([type]): [description]
            optimizer ([type]): [description]
            model (str, optional): [description]. Defaults to "Dense".
        """
        X_input = Input(input_shape)
        self.action_space = action_space

        # Shared CNN layers:
        if model=="CNN":
            X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)

        # Shared LSTM layers:
        elif model=="LSTM":
            X = LSTM(512, return_sequences=True)(X_input)
            X = LSTM(256)(X)

        # Shared Dense layers:
        else:
            X = Flatten()(X_input)
            X = Dense(512, activation="relu")(X)
        
        # Critic model
        V = Dense(512, activation="relu")(X)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

        # Actor model
        A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
        #print(self.Actor.summary())

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

    def critic_PPO2_loss(self, y_true, y_pred):
        """

            Loss function for Critic Network

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]

        Returns:
            [value_loss]: [description]
        """
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])








class Actor:
    def __init__(self,
                input_shape,
                action_space,
                lr,
                optimizer):
        
        self.input_shape = input_shape
        self.lr = lr
        self.optimizer = optimizer
        self.action_space = action_space
        self.actor = self.MLP()
        
    def summary(self):
        self.actor.summary()

    
    def MLP(self):
        X_input = Input(self.input_shape)
        X = Flatten(input_shape=self.input_shape)(X_input)
        X = Dense(512, activation="relu",       
                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)

        X = Dense(256, activation="relu",
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(128, activation="relu",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)

        output = Dense(self.action_space, activation="softmax")(X)

        actor = Model(inputs = X_input, outputs = output)
        actor.compile(loss=self.ppo_loss, optimizer = self.optimizer(lr=self.lr))

        return actor

    def ppo_loss(self, y_true, y_pred):
        """
        
        PPO uses a ratio between the newly updated policy and the old policy in the update step. 
        Computationally, it is easier to represent this in the log form: ratio = K.exp(K.log(prob) — K.log(old_prob));
        Using this ratio we can decide how much of a change in policy we are willing to tolerate.
        Hence, we use a clipping parameter LOSS_CLIPPING to ensure we only make the maximum percent change to our policy at a time.
        Paper suggests using this value as 0.2:
        p1=ratio*advantages
        p2=K.clip(ratio,min_value=1 — LOSS_CLIPPING, max_value=1+LOSS_CLIPPING)*advantages;
        Actor loss is calculating by simply getting a negative mean of element-wise minimum value of p1 and p2;
        Adding an entropy term is optional,
        but it encourages our actor model to explore different policies and the degree to which 
        we want to experiment can be controlled by an entropy beta parameter.
        
        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value = 1 - LOSS_CLIPPING, max_value = 1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = - (y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.actor.predict(state)


class Critic:
    def __init__(self, input_shape, action_space, lr, optimizer):
        self.input_shape = input_shape
        self.lr = lr
        self.action_space = action_space
        self.optimizer = optimizer

        self.critic = self.MLP()

    def summary(self):
        return self.critic.summary()

    def MLP(self):
        X_input = Input(self.input_shape)
        old_values = Input((1,))

        X = Flatten(input_shape=self.input_shape)(X_input)
        X = Dense(512, activation="relu",
                 kernel_initializer = 'he_uniform')(X)
        X = Dense(512, activation="relu",
                        kernel_initializer='he_uniform')(X)
        X = Dense(512, activation="relu",
        kernel_initializer='he_uniform')(X)

        value = Dense(1)(X)

        critic = Model(inputs = [X_input, old_values]  , outputs = value)

        critic.compile(loss=[self.ppo2_loss(old_values)], optimizer = self.optimizer(lr=self.lr))

        return critic

    def ppo2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.24
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2

            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))

            return value_loss
        return loss

    def predict(self,state):
        self.critic.predict([state, np.zeros((state.shape[0], 1))])


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = Flatten(input_shape=input_shape)(X_input)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(64, activation="relu")(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
        #print(self.Actor.summary)

    def summary(self):
        return self.Actor.summary()

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        # Arranging step size to be small:
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = - (y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)

        V = Flatten(input_shape=input_shape)(X_input)
        V = Dense(512, activation="relu")(V)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
