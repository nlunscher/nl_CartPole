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

class QL_Tree:
    MAX_REPLAY_STEPS = 20000
    RANDOM_ACTION_PROB = 0.1*3
    RANDOM_DECAY_RATE = 1.0/MAX_REPLAY_STEPS
    MAX_EPISODES = 1000*5
    MAX_ENV_STEPS = 1000
    QL_GAMMA = 0.9
    WATCH_INTERVAL = 20
    SAMPLE_TRAIN_SIZE = 10000
    FINAL_SCORE_PLACEHOLDER = 9999
    REMOVE_WORST = True

    def __init__(self, env_name):
        self.env = gym.make(env_name)

        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        print "Env:", env_name, "Observations:", self.obs_size, "Actions:", self.action_size

        self.replay_memory = [] # [state_obs, action, reward, next_state_obs, is_done, ep_num, final_score]

        self.agent = []
        for i in range(self.action_size):
            self.agent.append(DecisionTreeRegressor())
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

    def agent_play(self, show_game = True):
        observation = self.env.reset()
        total_steps = 0
        for t in range(1000):
            if show_game:
                self.env.render()
            action, predicted_Qs = self.agent_action(observation)
            new_observation, reward, done, info = self.env.step(action)
            print predicted_Qs
            if done:
                total_steps = t + 1
                if show_game:
                    print("Agent finished after {} timesteps".format(total_steps))
                break

            observation = new_observation

            if show_game:
                time.sleep(0.05)
        return total_steps

    def agent_action(self, observation):
        if self.agent_is_trained:
            predicted_Qs = []
            for i in range(self.action_size):
                predicted_Qs.append(self.agent[i].predict([observation]))
            # print predicted_Qs
            action = np.argmax(predicted_Qs)
        else:
            predicted_Qs = np.zeros(self.action_size)
            action = self.env.action_space.sample()
        return action, predicted_Qs

    def train_agent(self):
        # print "Training Agent"
        batch_memory = random.sample(self.replay_memory, min(len(self.replay_memory), self.SAMPLE_TRAIN_SIZE))
        batch_size = len(batch_memory)

        max_predicted_Qs = np.zeros(batch_size)
        for i in range(batch_size):
            action, qs = self.agent_action(batch_memory[i][3])
            max_predicted_Qs[i] = np.max(qs)

        max_target_Q = np.zeros(batch_size)
        target_Qs = []
        features = []
        for i in range(self.action_size):
            features.append([])
            target_Qs.append([])
        for i in range(batch_size):
            state_obs, action, reward, next_state_obs, is_done, ep_num, final_score = batch_memory[i]

            max_target_Q[i] = reward 
            if not is_done:
                max_target_Q[i] += self.QL_GAMMA * max_predicted_Qs[i]
            target_Qs[action].append(max_target_Q[i])
            features[action].append(state_obs)

        # features = [f[0] for f in batch_memory]

        # print "showing"
        for i in range(self.action_size):
            self.agent[i].fit(features[i], target_Qs[i])
            # print "Agent Score", self.agent[i].score(features[i], target_Qs[i])

        # # print features[0]
            # print "-targ", target_Qs[i][:5]
            # print "pred", self.agent[i].predict(features[i])[:5]

        if not self.agent_is_trained:
            self.agent_is_trained = True
            print "Trained"

    def train(self):
        agent_steps = []
        for ep in range(self.MAX_EPISODES):
            if ep % self.WATCH_INTERVAL == 0:
                # watch the agent play
                total_avg = 0
                for i in range(10):
                    total_avg += self.agent_play(show_game=False)
                total_avg /= 10.0
                agent_steps.append(total_avg)
                print "Average", total_avg
                total_steps = self.agent_play()

            obs = self.env.reset()
            total_reward = 0
            for step in range(self.MAX_ENV_STEPS):
                # self.env.render()

                decayed_random_prob = self.RANDOM_ACTION_PROB * np.exp(-1*self.RANDOM_DECAY_RATE*ep)
                if random.random() < decayed_random_prob:
                    action = self.env.action_space.sample()
                else:
                    action,_ = self.agent_action(obs)

                new_obs, reward, is_done, info = self.env.step(action)

                if is_done:
                    reward = -100
                total_reward += reward

                self.replay_memory.append([obs, action, reward, new_obs, is_done, ep, self.FINAL_SCORE_PLACEHOLDER])
                if len(self.replay_memory) > self.MAX_REPLAY_STEPS:
                    if self.REMOVE_WORST:
                        lowest_final_index = 0
                        min_final = self.FINAL_SCORE_PLACEHOLDER
                        max_final = -self.FINAL_SCORE_PLACEHOLDER
                        for i in range(len(self.replay_memory)):
                            if self.replay_memory[i][-1] < min_final:
                                min_final = self.replay_memory[i][-1]
                                lowest_final_index = i
                            if self.replay_memory[i][-1] > max_final and self.replay_memory[i][-1] != self.FINAL_SCORE_PLACEHOLDER:
                                max_final = self.replay_memory[i][-1]
                        popped = self.replay_memory.pop(lowest_final_index)
                    else:
                        self.replay_memory.pop(0)

                    if ep % 10 == 0 and False:
                        print "Ep", ep, "step", step, "Popped", popped[-3:], "Max", max_final, \
                            "Test", agent_steps[-1]

                if is_done or step == self.MAX_ENV_STEPS - 1:
                    for i in range(len(self.replay_memory)):
                         if self.replay_memory[i][-1] == self.FINAL_SCORE_PLACEHOLDER:
                            self.replay_memory[i][-1] = total_reward
                    break

                obs = new_obs
                # time.sleep(0.1)

            if len(self.replay_memory) >= 100:
                self.train_agent()

        plt.plot(range(self.MAX_EPISODES/self.WATCH_INTERVAL), agent_steps)
        plt.show()


if __name__ == "__main__":
    ql = QL_Tree('CartPole-v0')
    # ql.random_play()

    ql.train()