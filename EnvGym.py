import gym

# env = gym.make("MountainCar-v0")
env = gym.make('CartPole-v0')
env.reset()

for i in range(100):
    # action = 0 or 1
    action = env.action_space.sample()

    # env.step(0)
    # env.step(1)
    env.step(action)

    env.render()

env.close()