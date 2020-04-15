import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps) / episodes

    file = open("actions", "w")

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        timesteps = 0

        file.write("\nEpisode #{0} : ".format(i))

        while done != True:

            timesteps = timesteps + 1

            # Render environment for last five episodes
            if i >= (episodes - 20):
                # print('\n\nEpisode {} Average Reward: {}'.format(i + 1, ave_reward))
                env.render()

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # print("{0} ".format(action))
            file.write("{0}, ".format(action))

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            if(reward > -1.0 or reward < -1.0):
                print(reward)

            # if i == 4999:
            #     print('\n\nEpisode {} Average Reward: {}'.format(i + 1, reward))

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
            # Adjust Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state2_adj[0],
                                                        state2_adj[1]]) -
                                    Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Track rewards
        reward_list.append(tot_reward)
        file.write("\ntot_reward {0}".format(tot_reward))

        if (i + 1) % 500 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

        # f.write("\Reward : " + )

    file.close()
    env.close()

    return ave_reward_list


# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 5000)

# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')
plt.close()
