import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class QLearningAgent:
    def __init__(self, env, alpha=0.2, gamma=0.9, epsilon=0.9, iterations=5000, Q=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.iterations = iterations
        self.min_eps = 0
        self.Q = Q
        self.rewards = []

    def __create_q_table(self, x, y, action_count):
        """
        Method for creating a Q table of with x by y for n actions.
        """
        return np.zeros((x, y, action_count))

    def __get_Q_state(self, env_state):
        """
        Method for finding a Q state given an environment state
        """
        env_x, env_y = env_state
        q_x, q_y, _ = self.Q.shape

        diff = abs(self.env.observation_space.high -
                   self.env.observation_space.low)
        step_size_x = diff[0] / q_x
        step_size_y = diff[1] / q_y

        return [int(env_x / step_size_x), int(env_y / step_size_y)]

    def __get_next_action(self, action_list):
        """
        Method for getting the next action given a current Q state
        """
        if np.random.random() < 1 - self.epsilon:
            return np.argmax(action_list)

        return np.random.randint(0, self.env.action_space.n)

    def __get_max_value(self, state):
        """
        Method for getting the max action given a state
        """
        return np.max(self.Q[state[0], state[1]])

    def __calc_new_value(self, reward, state, action, next_state):
        """
        Method for calculating the next Q value
        """
        return self.Q[state[0], state[1], action] + self.alpha * (reward + self.gamma * self.__get_max_value(next_state) - self.Q[state[0], state[1], action])

    def save(self, location):
        """
        Method for saving the Q table
        """
        with open(location, 'wb') as loc:
            pickle.dump(self.Q, loc, pickle.HIGHEST_PROTOCOL)

    def load(self, location):
        """
        Method for loading the Q table
        """
        try:
            with open(location, 'rb') as loc:
                self.Q = pickle.load(loc)
        except:
            print("Something went wrong")
            return

    def play_environment(self):
        """
        Method for playing according to a Q table without updating it
        """
        # if not self.Q:
        #     return

        state = env.reset()
        q_state = self.__get_Q_state(state[0])
        done = False
        while not done:
            action = self.__get_next_action(self.Q[q_state[0]][q_state[1]])
            next_state, reward, done, _, _ = env.step(action)
            next_q_state = self.__get_Q_state(next_state)

            if done and next_state[0] >= 0.5:
                return

            q_state = next_q_state

    def __plot_rewards(self):
        print(self.rewards)

    def q_learning(self):
        if not self.Q:
            self.Q = self.__create_q_table(20, 200, env.action_space.n)

        for _ in tqdm(range(self.iterations)):
            state = env.reset()
            q_state = self.__get_Q_state(state[0])
            done = False
            tot_reward, reward = 0, 0
            while not done:
                action = self.__get_next_action(self.Q[q_state[0]][q_state[1]])
                next_state, reward, done, _, _ = env.step(action)
                next_q_state = self.__get_Q_state(next_state)

                if done and next_state[0] >= 0.5:
                    self.Q[q_state[0], q_state[1], action] = reward
                else:
                    self.Q[q_state[0], q_state[1], action] = self.__calc_new_value(
                        reward, q_state, action, next_q_state)

                tot_reward += reward
                q_state = next_q_state

            self.rewards.append(tot_reward)
            self.epsilon = self.epsilon - 2 / self.iterations if self.epsilon > 0.01 else 0.01


if __name__ == '__main__':
    env_type = input("Type T for train and P for play: ")

    if env_type == 'T':

        env = gym.make('MountainCar-v0')
        env.reset()
        agent = QLearningAgent(env, 0.2, 0.9, 0.8, 5000)
        agent.q_learning()
        agent.save('saved/q_learning.bson')

    elif env_type == 'P':
        env = gym.make('MountainCar-v0', render_mode="human")
        agent = QLearningAgent(env)
        agent.load('saved/q_learning.bson')
        agent.play_environment()
