from agent import DQNAgent
from train import mini_batch_train
# from env import env_example
from transmit_data_env import IoTCommunicationEnv
# import gym

MAX_EPISODES = 500
MAX_STEPS = 500
BATCH_SIZE = 256

# env_id = "CartPole-v0"

num_user = 4
number_power = 3
max_power = 0.0316227766
max_channel = 3

env = IoTCommunicationEnv(num_user, number_power, max_power, max_channel)
# env = gym.make(env_id)
# env = env_example(3, 2, 3)
# env.seed(1)

agent = DQNAgent(env, buffer_size=20000)
print("x,y")
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)