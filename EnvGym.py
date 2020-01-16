import gym

# env = gym.make("MountainCar-v0")
env = gym.make('CartPole-v0')
env.reset()

for episodes in range(10):
    observation = env.reset()
    for timesteps in range(100):
        env.render()
        # action = 0 or 1
        action = env.action_space.sample()

        # env.step(0)
        # env.step(1)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode #{0} ended after {1} timesteps".format(episodes, timesteps))
            break
env.close()