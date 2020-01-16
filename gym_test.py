import gym


# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(100):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()


# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     print("i_episode : {}".format(i_episode))
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

num1 = 12
num2 = 13.666

# env = gym.make('CartPole-v0')
env = gym.make('SpaceInvaders-v0')
print(env.action_space)
lol = env.action_space

#> Discrete(2)
print(env.observation_space)
#> Box(4,)

env2 = gym.make('MountainCar-v0')
print(env2.action_space.__repr__())
print(env2.action_space.n)

