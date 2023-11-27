from agent import DQNAgent
from train import mini_batch_train
from env import env_example

MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

env = env_example(3, 2, 3)
agent = DQNAgent(env)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)