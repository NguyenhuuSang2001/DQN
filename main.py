from agent import DQNAgent
from train import mini_batch_train
from env import env_example

MAX_EPISODES = 10000
MAX_STEPS = 100
BATCH_SIZE = 128

env = env_example(3, 2, 3)
env.seed(1)

agent = DQNAgent(env)
print("x,y")
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)