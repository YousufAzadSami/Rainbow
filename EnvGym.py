import gym
import torch
import numpy as np


def playgroundGym():
    # env = gym.make("MountainCar-v0")
    env = gym.make('CartPole-v0')
    env.reset()

    # Debug to find the data type
    # Ctrl + , to find the class
    actions = env.action_space
    # The variable type of env.observation_space and the one we get from env.step() is different
    observations = env.observation_space
    print(observations)

    observations2, _, _, _ = env.step(env.action_space.sample())
    converted = torch.from_numpy(observations2)
    print(converted)

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

# playgroundGym()

def playgroundTorch():
    tor = torch.rand(3, 4)
    print(tor)
    print(type(tor))
    print(tor.type())
    print(tor.size())

    tor = torch.rand(3)
    print(tor.size())

    # print(observation)
    # print(type(observation))
    # print(observation.shape())

    # region result
    # tensor([[0.5455, 0.6471, 0.6017, 0.2713],
    #         [0.5937, 0.2467, 0.4359, 0.2361],
    #         [0.2094, 0.9217, 0.5257, 0.9479]])
    # <
    #
    # class 'torch.Tensor'>
    #
    # torch.FloatTensor
    # torch.Size([3, 4])
    # torch.Size([3])
    # tensor([[0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
    #         [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
    #         ...,
    #         [0.3098, 0.3098, 0.3098, ..., 0.3098, 0.3098, 0.3098],
    #         [0.3098, 0.3098, 0.3098, ..., 0.3098, 0.3098, 0.3098],
    #         [0.3098, 0.3098, 0.3098, ..., 0.3098, 0.3098, 0.3098]])
    # <
    #
    # class 'torch.Tensor'>
    #
    # torch.Size([84, 84])
    # endregion

    num = np.arange(20).reshape(4, 5)
    print(num)
    print(type(num))

    # torch.Size([4, 5])
    converted = torch.from_numpy(num)
    print(converted)
    print(type(converted))

    convertedFloat = converted.float()
    print(type(convertedFloat))



# playgroundTorch()

class EnvGym():
    def __init__(self, game_name):
        self.envGym = gym.make(game_name)

    def reset(self):
        observation = self.envGym.reset()
        # print(type(observation))
        return torch.from_numpy(observation).float()

    def step(self, action):
        # action = self.envGym.action_space.sample()
        observation, reward, done, _ = self.envGym.step(action)
        return torch.from_numpy(observation).float(), reward, done
    def action_space(self):
        return self.envGym.action_space.n

# envGym = EnvGym("MountainCar-v0")
# observation = envGym.reset()
# observation_con01 = observation.float()
# observation_con02 = torch.as_tensor(observation, dtype=torch.float32)
# print(observation_con02)


# tensor([-0.4256,  0.0000])
# torch.Size([2])
# tensor(0, dtype=torch.uint8)
#
#
#
# tensor([0, 1])
# torch.Size([2])
# tensor(255)
# tensor(255, dtype=torch.uint8)