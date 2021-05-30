import pandas as pd, numpy as np
from datetime import datetime
from collections import deque



def discretize_action(action):  
    if 0.1 > action > - 0.1:
        action = 0 # Hold
    elif action >= 0.1:
        action = 1 # Buy
    else:
        action = 2 # Sell
    return action




def train(env, agent, visualize=False, train_episodes = 50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_obs, train_episodes) # create TensorBoard writer
    total_average = deque(maxlen=360) # save recent 360 episodes net worth
    best_average = 0 # used to track best average net worth

    print('Training starts...')

    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = agent.act(state)
            next_state, reward, done , info = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state


        actor_loss, critic_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)
        
        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
        agent.writer.add_scalar('Data/actor_loss', actor_loss, episode)
        agent.writer.add_scalar('Data/critic_loss', critic_loss, episode)
        
        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average),
                           args=[episode, average, env.episode_orders, actor_loss, critic_loss])
            agent.save()
            
    agent.end_training_log()






def train_continious(env, agent, visualize=False, train_episodes = 50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_obs, train_episodes) # create TensorBoard writer
    total_average = deque(maxlen=360) # save recent 360 episodes net worth
    best_average = 0 # used to track best average net worth

    print('Training starts...')

    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = agent.act(state)
            next_state, reward, done , info = env.step(prediction)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)

            # Tanh:
            action = discretize_action(action)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        actor_loss, critic_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)
        print('Test 1')
        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
        agent.writer.add_scalar('Data/actor_loss', actor_loss, episode)
        agent.writer.add_scalar('Data/critic_loss', critic_loss, episode)
        print('Test 2')
        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(episode, env.net_worth, average, env.episode_orders))
        print('Test 3')
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average),
                           args=[episode, average, env.episode_orders, actor_loss, critic_loss])
            agent.save()
            
    agent.end_training_log()
