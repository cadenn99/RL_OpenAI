import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from models.Agent import Agent


class QLearningAgent(Agent):

    def __init__(self, env, alpha=0.2, gamma=0.9, epsilon=0.9, iterations=5000, Q=None):
        super().__init__(env, alpha, gamma, epsilon, iterations, Q)

    def __calc_new_value(self, reward, state, action, next_state):
        """
        Method for calculating the next Q value
        """
        return self.Q[state[0], state[1], action] + self.alpha * (reward + self.gamma * self._get_max_value(next_state) - self.Q[state[0], state[1], action])

    def train(self):
        """
        Method for training the model
        """
        if self.Q == None:
            self.Q = self._create_q_table(20, 200, self.env.action_space.n)

        for _ in tqdm(range(self.iterations), ncols=100):
            state = self.env.reset()
            q_state = self._get_Q_state(state[0])
            done = False

            iteration_rewards = 0
            while not done:
                action = self._get_next_action(
                    self.Q[q_state[0]][q_state[1]])
                next_state, reward, done, _, _ = self.env.step(action)
                next_q_state = self._get_Q_state(next_state)

                if done and next_state[0] >= 0.5:
                    self.Q[q_state[0], q_state[1], action] = reward
                else:
                    self.Q[q_state[0], q_state[1], action] = self.__calc_new_value(
                        reward, q_state, action, next_q_state)

                iteration_rewards += reward
                q_state = next_q_state

            self.rewards.append(iteration_rewards)
            self.epsilon = self.epsilon - 2 / self.iterations if self.epsilon > 0.01 else 0.01
        self.env.close()

        self._plot_rewards('Q', self.iterations)
