import gym
import torch
import numpy as np


def playgroundGym():
    env = gym.make("MountainCar-v0")
    # env = gym.make('CartPole-v0')
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

            print("#{0} timestep : {1}".format(timesteps, observation))

            if done:
                print("Episode #{0} ended after {1} timesteps\n\n".format(episodes, timesteps))
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

# region numpy_indexing_slicing
# np.arange(10)
# > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# np.arange(10).reshape(2,5)
# > array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
# np.arange(10).reshape(2,-1)
# > array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
# np.arange(10).reshape(2,-2)
# > array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
# np.arange(10).reshape(2,-3)
# > array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
# x = np.arange(10).reshape(2,-3)
# x
# > array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
# x[-1]
# > array([5, 6, 7, 8, 9])
# x = np.arange(15).reshape(3,-3)
# x
# > array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
# np.arange(15).reshape(3,-3)
# > array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
# np.arange(15).reshape(3,-1)
# > array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
# np.arange(15).reshape(3,-5)
# > array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
# x = np.arange(15).reshape(3,-1)
# x
# > array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
# x[-1]
# > array([10, 11, 12, 13, 14])
# x[-1]
# > array([10, 11, 12, 13, 14])
# x[-2]
# > array([5, 6, 7, 8, 9])
# x[-3]
# > array([0, 1, 2, 3, 4])
# x[-4]
# Traceback (most recent call last):
#   File "<input>", line 1, in <module>
# IndexError: index -4 is out of bounds for axis 0 with size 3
# y = np.arange(10)
# y
# > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# y[-2]
# > 8
# y[1:5]
# > array([1, 2, 3, 4])
# y[0:5]
# > array([0, 1, 2, 3, 4])
# y[-1:5]
# > array([], dtype=int64)
# y[-1:-5]
# > array([], dtype=int64)
# y[1:-5]
# > array([1, 2, 3, 4])
# y[1:-2]
# > array([1, 2, 3, 4, 5, 6, 7])
# y[1:-2]
# > array([1, 2, 3, 4, 5, 6, 7])
# y[1:6:2]
# > array([1, 3, 5])
# np.arange(24).reshape(2, 3, 4)
# > array([[[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]],
#        [[12, 13, 14, 15],
#         [16, 17, 18, 19],
#         [20, 21, 22, 23]]])
# z = np.arange(24).reshape(2, 3, 4)
# z[-1]
# > array([[12, 13, 14, 15],
#        [16, 17, 18, 19],
#        [20, 21, 22, 23]])
# z[1, -1]
# > array([20, 21, 22, 23])
# z[0, -1]
# > array([ 8,  9, 10, 11])

# endregion

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

# y = torch.tensor(range(20))
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
#         18, 19])
# y = y.view(2, 10)
# tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
# y = y.view(2, -1) // -1 means, Python will figure out that number
# tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
# y[:, 3:6]
# tensor([[ 3,  4,  5],
#         [13, 14, 15]])
# y[1, 3:6]
# tensor([13, 14, 15])
# y[1, range(3,6)]
# tensor([13, 14, 15])