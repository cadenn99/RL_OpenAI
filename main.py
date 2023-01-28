from models.QLearningAgent import QLearningAgent
from models.SARSALearningAgent import SARSALearningAgent
import gym


def main():
    model = input("Type S for SARSA Learning or Q for Q Learning: ")
    env_type = input("Type T for train and P for play: ")

    if env_type == 'T':

        env = gym.make('MountainCar-v0')
        env._max_episode_steps = 1000
        agent = QLearningAgent(env, 0.2, 0.9, 0.8, 5000) if model.lower(
        ) == 'q' else SARSALearningAgent(env, 0.2, 0.9, 1, 5000)
        agent.train()
        file = 'data/sarsa_learning.bson' if model.lower() == 's' else 'data/q_learning.bson'
        agent.save(file)

    elif env_type == 'P':
        env = gym.make('MountainCar-v0', render_mode="human")
        agent = QLearningAgent(env, epsilon=0) if model.lower(
        ) == 'q' else SARSALearningAgent(env, epsilon=0)
        file = 'data/sarsa_learning.bson' if model.lower() == 's' else 'data/q_learning.bson'
        agent.load(file)
        agent.play_environment()


if __name__ == '__main__':
    main()
