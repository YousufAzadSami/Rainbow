import gym
import torch
import numpy as np
# import readchar


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

    observation, _, _, _ = env.step(env.action_space.sample())
    observation_torch = torch.from_numpy(observation)
    print(observation_torch)
    # this output is in the format of env.py output of states
    observation_stack = torch.stack([observation_torch, observation_torch, observation_torch])
    # observation_stack_stack = torch.stack([observation_stack, observation_stack])
    print("end 01")

    var = []
    var.append(observation_torch)
    var.append(observation_torch)
    var.append(observation_torch)
    # this output is in the format of env.py output of states
    observation_stack2 = torch.stack(var)
    print("end 02")


    for episodes in range(10):
        observation = env.reset()
        rewards = 0
        for timesteps in range(100):
            env.render()
            # action = 0 or 1
            action = env.action_space.sample()

            # The variable type of env.observation_space and the one we get from env.step() is different
            # env.step(0)
            # env.step(1)
            observation, reward, done, info = env.step(action)

            rewards = rewards + reward

            print("#{0} Reward : {2}, Timestep : {1}".format(timesteps, observation, rewards))

            if done:
                print("Episode #{0} ended after {1} timesteps\n\n".format(episodes, timesteps))
                break
    env.close()

# playgroundGym()




def playgroundGymFresh():
    env = gym.make('MountainCar-v0')  # try for different environements
    env.reset()

    # actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # push left
    # actions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # no push
    # actions = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]   # push right
    actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]

    for timesteps in range(200):
        env.render()
        action = actions[timesteps]
        timesteps = timesteps + 1

        observation, reward, done, info = env.step(action)
        print("Timestep : {0:3d}, Action: {1}, Reward: {2}".format(timesteps, action, reward))

        if done:
            print("Finished after {} timesteps".format(timesteps + 1))
            break
    env.close()

playgroundGymFresh()





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




# envGym = EnvGym("MountainCar-v0")
# observation_1 = envGym.reset()
#
# observation_2, reward, done = envGym.step(2)
# print("end")

# IMPORTANT
# Have a look at this to under stand the tensor size and it's dimensions
# a = torch.tensor(2)
# a.shape
# torch.Size([])
# b = torch.tensor([44])
# b.shape
# torch.Size([1])
# c = torch.tensor([1, 3])
# c.shape
# torch.Size([2])
# d = torch.tensor([1, 3, 5, 7])
# d.shape
# torch.Size([4])
# e = torch.tensor([[111, 33, 55, 5], [12, 12, 23, 45]])
# e
# tensor([[111,  33,  55,   5],
#         [ 12,  12,  23,  45]])
# e.shape
# torch.Size([2, 4])
# f = torch.tensor([[[111, 33, 55, 5], [12, 12, 23, 45]], [[111, 33, 55, 5], [12, 12, 23, 45]], [[111, 33, 55, 5], [12, 12, 23, 45]]])
# f.shape
# torch.Size([3, 2, 4])


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