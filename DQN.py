import gym
import numpy as np
import time
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model

env = gym.make('LunarLander-v2')
env.seed(1213)

class Memory:
    def __init__(self):
        self.max_size = 1000000
        self.counter = 0
        self.old_state = np.zeros((self.max_size, env.observation_space.shape[0]))
        self.action = np.zeros(shape=self.max_size,dtype=int)
        self.reward = np.zeros((self.max_size, 1))
        self.new_state = np.zeros((self.max_size, env.observation_space.shape[0]))

    def add_to_memory(self, pre_observation, action, reward, next_observation):
        index = self.counter % self.max_size
        self.old_state[index] = pre_observation
        self.action[index] = action
        self.reward[index] = reward
        self.new_state[index] = next_observation
        self.counter += 1


memory = Memory()
# trainning model
model = Sequential([Dense(250, activation='relu'), Dense(
    250, activation='relu'), Dense(env.action_space.n, activation='linear')])
model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
# target_model
target_model = Sequential([Dense(250, activation='relu'), Dense(
    250, activation='relu'), Dense(env.action_space.n, activation='linear')])
target_model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])


# get random action or predict from model
epsilon = 1.00


def choose_action(observation):
    observation_ = observation[np.newaxis, :]
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(observation_))


def train_model():
    if memory.counter >= 1000:
        old_state, reward, action, new_state = sample_buffer()
        q_value_target = target_model.predict(new_state)
        q_replace = model.predict(old_state)
        indices = np.arange(64)
        q_replace[indices,action] = np.max(0.85*q_value_target+reward,axis=1)
        model.train_on_batch(old_state, q_replace)
        global epsilon
        if epsilon > 0.01:
            epsilon *= 0.999
        if memory.counter % 100 == 0:
            target_model.set_weights(model.get_weights())
            model.save('DQN_predict_model.h5')
            target_model.save('DQN_target_model.h5')


def sample_buffer():
    batch_index_list = np.random.choice(
        min(memory.counter, memory.max_size-1), 64)
    old_state = memory.old_state[batch_index_list]
    reward = memory.reward[batch_index_list]
    new_state = memory.new_state[batch_index_list]
    action = memory.action[batch_index_list]
    return old_state, reward, action, new_state


for i_episode in range(1000000):

    observation = env.reset()
    total_r = 0
    while True:
        env.render()
        action = choose_action(observation)
        new_observation, reward, done, info = env.step(action)
        total_r += reward
        memory.add_to_memory(observation, action, reward,
                             new_observation)
        observation = new_observation
        train_model()
        if done:
            break
    print('Episode:', i_episode, ' Total Reward:', total_r)
env.close()
