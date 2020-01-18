import gym
import torch

print("yup!")


def playground():
    # env = gym.make("MountainCar-v0")
    env = gym.make('CartPole-v0')
    env.reset()

    # Debug to find the data type
    # Ctrl + , to find the class
    actions = env.action_space
    # The variable type of env.observation_space and the one we get from env.step() is different
    observations = env.observation_space
    print(observations)

    for episodes in range(10):
        observation = env.reset()
        for timesteps in range(100):
            env.render()
            # action = 0 or 1
            action = env.action_space.sample()

            # The variable type of env.observation_space and the one we get from env.step() is different
            # env.step(0)
            # env.step(1)
            observation, reward, done, info = env.step(action)

            print(observation)

            if done:
                print("Episode #{0} ended after {1} timesteps".format(episodes, timesteps))
                break
    env.close()


class EnvGym:
    def __init__(self):
        self.envGym = gym.make("MountainCar-v0")

    def reset(self):
        observation = self.envGym.reset()
        print(type(observation))

    def step(self):
        action = self.envGym.action_space.sample()
        observation, reward, done, info = self.envGym.step(action)


env_gym = EnvGym()
env_gym.reset()
env_gym.step()

