from tqdm import tqdm
from models.Agent import Agent


class SARSALearningAgent(Agent):

    def __init__(self, env, alpha=0.2, gamma=0.9, epsilon=0.9, iterations=5000, Q=None):
        super().__init__(env, alpha, gamma, epsilon, iterations, Q)

    def __calc_new_value(self, reward, state, action, next_state, next_action):
        """
        Method for calculating the next Q value
        """
        return self.Q[state[0], state[1], action] + self.alpha * (reward + self.gamma * self.Q[next_state[0], next_state[1], next_action] - self.Q[state[0], state[1], action])

    def train(self):
        """
        Method for training the model
        """
        if self.Q == None:
            self.Q = self._create_q_table(20, 20, self.env.action_space.n)

        for _ in tqdm(range(self.iterations), ncols=100):
            state = self.env.reset()
            q_state = self._get_Q_state(state[0])
            action = self._get_next_action(q_state)

            done = False
            trunc = False
            iteration_rewards = 0
            while not done and not trunc:
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_q_state = self._get_Q_state(next_state)
                next_action = self._get_next_action(next_q_state)

                self.Q[q_state[0], q_state[1], action] = self.__calc_new_value(
                    reward, q_state, action, next_q_state, next_action)

                if done:
                    self.Q[q_state[0], q_state[1], action] = reward

                iteration_rewards += reward
                q_state, action = next_q_state, next_action

            self.rewards.append(iteration_rewards)

            self.epsilon = self.epsilon - 2 / self.iterations if self.epsilon > 0.01 else 0.01

        self.env.close()

        self._plot_rewards('SARSA', self.iterations)
