import gym
import torch
import numpy as np



class EnvGym():
    def __init__(self, game_name):
        self.envGym = gym.make(game_name)
        self.frame_num = 4
        self.done_counter = 0
        self.reached_flag = 0
        self.file_local = None

    def reset_2(self):
        list_observations = []
        for i in range(self.frame_num):
            observation = self.envGym.reset()
            observation_2x2 = np.array([observation, [1, 1]])
            list_observations.append(torch.from_numpy(observation_2x2).float())
        return torch.stack(list_observations)

    # original - [4 , 84, 84]
#                [2, 1] > [4, 2, 2]
    def step_2(self, action):
        list_observations = []
        rewards = 0
        for i in range(self.frame_num):
            observation, reward, done, _ = self.envGym.step(action)
            observation_2x2 = np.array([observation, [1, 1]])
            list_observations.append(torch.from_numpy(observation_2x2).float())
            # print("#{0} observation : {1}".format(i, observation))
            rewards = rewards + reward
            # TODO : What to do about 'done'
        return torch.stack(list_observations), rewards, done

    def set_file(self, file):
        self.file_local = file

    def reset(self):
        observation = self.envGym.reset()
        # print(type(observation))
        return torch.from_numpy(observation).float()

    def step(self, action):
        # action = self.envGym.action_space.sample()

        # for i in range(20):
        observation, reward, done, _ = self.envGym.step(action)

        if done is True and observation[0] >= 0.5:
            if self.done_counter >= 5:
                self.done_counter = 0
                self.reached_flag = self.reached_flag + 1
                print("{0} : Agent is `done` for 5 steps".format(self.reached_flag))
                if self.file_local is not None:
                    self.file_local.write("\n{0} : Agent is `done` for the 5th step".format(self.reached_flag))
                # break
            else:
                done = False
                reward = 0
                self.done_counter = self.done_counter + 1

        return torch.from_numpy(observation).float(), reward, done

    def action_space(self):
        return self.envGym.action_space.n

    def close(self):
        self.envGym.close()

    def render(self):
        self.envGym.render()
