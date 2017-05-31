# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print "Obs", observation
#         action = env.action_space.sample()
#         new_observation, reward, done, info = env.step(action)
#         print "Action", action, "Reward", reward, "Done", done, "Info", info
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

#         observation = new_observation

# reference https://gist.github.com/viswanathgs/ca9788020cfcc7849b9181d9239a2ef4

# [position of cart, velocity of cart, angle of pole, rotation rate of pole]

import gym
import numpy as np
import time
import random
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

class QL:
    MAX_REPLAY_STEPS = 1000
    RANDOM_ACTION_PROB = 0.5
    MAX_EPISODES = 100
    MAX_ENV_STEPS = 1000
    QL_GAMMA = 0.8

    def __init__(self, env_name):
        self.env = gym.make(env_name)

        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        print "Env:", env_name, "Observations:", self.obs_size, "Actions:", self.action_size

        self.replay_memory = [] # [state_obs, action, reward, next_state_obs]

        self.agent = DecisionTreeRegressor()
        self.agent_is_trained = False

    def random_play(self):
        observation = self.env.reset()
        for t in range(1000):
            self.env.render()
            print "Obs", observation
            action = self.env.action_space.sample()
            new_observation, reward, done, info = self.env.step(action)
            print "Action", action, "Reward", reward, "Done", done, "Info", info
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

            observation = new_observation

            time.sleep(0.1)

    def agent_play(self):
        observation = self.env.reset()
        total_steps = 0
        for t in range(1000):
            self.env.render()
            action = self.agent_action(observation)
            new_observation, reward, done, info = self.env.step(action)
            if done:
                total_steps = t + 1
                print("Agent finished after {} timesteps".format(total_steps))
                break

            observation = new_observation

            time.sleep(0.1)
        return total_steps

    def agent_action(self, observation):
        if self.agent_is_trained:
            predicted_Qs = self.agent.predict([observation])
            action = np.argmax(predicted_Qs)
        else:
            action = self.env.action_space.sample()
        return action

    def train_agent(self):
        # print "Training Agent"

        max_predicted_Qs = np.zeros(len(self.replay_memory))
        for i in range(len(self.replay_memory)):
            qs = self.agent_action(self.replay_memory[i][0])
            max_predicted_Qs[i] = np.argmax(qs)

        max_target_Q = np.zeros(len(self.replay_memory))
        target_Qs = np.zeros((len(self.replay_memory), self.action_size))
        for i in range(len(self.replay_memory)):
            state_obs, action, reward, next_state_obs = self.replay_memory[i]

            max_target_Q[i] = reward + self.QL_GAMMA * max_predicted_Qs[i]
            target_Qs[i][action] = max_target_Q[i]

        features = [f[0] for f in self.replay_memory]

        self.agent.fit(features, target_Qs)
        # print "Agent Score", self.agent.score(features, target_Qs)
        self.agent_is_trained = True

    def train(self):
        agent_steps = []
        for ep in range(self.MAX_EPISODES):
            if ep % 10 == 0:
                # watch the agent play
                total_steps = self.agent_play()
                agent_steps.append(total_steps)

            obs = self.env.reset()
            for step in range(self.MAX_ENV_STEPS):
                # self.env.render()

                if random.random() < self.RANDOM_ACTION_PROB:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent_action(obs)

                new_obs, reward, is_done, info = self.env.step(action)

                if is_done:
                    reward = -100

                self.replay_memory.append([obs, action, reward, new_obs])
                if len(self.replay_memory) > self.MAX_REPLAY_STEPS:
                    self.replay_memory.pop(0)

                if is_done:
                    # print("Episode finished after {} timesteps".format(step+1))
                    break

                obs = new_obs
                # time.sleep(0.1)

            if len(self.replay_memory) > 0:
                self.train_agent()

        plt.plot(range(self.MAX_EPISODES/10), agent_steps)
        plt.show()


if __name__ == "__main__":
    ql = QL('CartPole-v0')
    # ql.random_play()

    ql.train()