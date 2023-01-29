import numpy as np
import matplotlib.pyplot as plt
import pickle


class Agent:
    def __init__(self, env, alpha=0.2, gamma=0.9, epsilon=1, iterations=5000, Q=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.iterations = iterations
        self.Q = Q
        self.rewards = []

    def _get_next_action(self, state):
        """
        Method for getting the next action given a current Q state
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.env.action_space.n)

        return np.argmax(self.Q[state[0], state[1]])

    def _create_q_table(self, x, y, action_count):
        """
        Method for creating a Q table of with x by y for n actions.
        """
        return np.zeros((x, y, action_count))

    def _get_Q_state(self, env_state):
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

    def _get_max_action(self, state):
        """
        Method for getting the max action given a state
        """

        return np.argmax(self.Q[state[0], state[1]])

    def moving_average(self, a):
        """
        Credits: Jaime

        https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
        """
        n = int(len(a) / 100)
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

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

    def _plot_rewards(self, model, episodes):
        """
        Method for plotting the rewards over time
        """
        average = self.moving_average(self.rewards)
        plt.plot(range(0, len(average)), average)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.savefig('{}_{}.jpg'.format(model, episodes))
        plt.close()

    def _track_stats(self):
        pass

    def play_environment(self):
        """
        Method for playing according to a Q table without updating it
        """
        if self.Q.all() == None:
            return

        state = self.env.reset()
        q_state = self._get_Q_state(state[0])
        done = False
        while not done:
            action = self._get_next_action(q_state)
            next_state, reward, done, _, _ = self.env.step(action)
            next_q_state = self._get_Q_state(next_state)

            if done and next_state[0] >= 0.5:
                return

            q_state = next_q_state
