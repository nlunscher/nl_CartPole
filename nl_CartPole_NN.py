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

# from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf

class QL_NN:
    MAX_REPLAY_STEPS = 1000*10
    RANDOM_ACTION_PROB = 0.1*5
    MAX_EPISODES = 1000
    MAX_ENV_STEPS = 1000
    QL_GAMMA = 0.9
    WATCH_INTERVAL = 10
    SAMPLE_TRAIN_SIZE = 128
    LEARNING_RATE = 0.001

    def __init__(self, env_name, sess):
        self.env = gym.make(env_name)
        self.sess = sess

        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        print "Env:", env_name, "Observations:", self.obs_size, "Actions:", self.action_size

        self.replay_memory = [] # [state_obs, action, reward, next_state_obs, is_done]

        self.agent_is_trained = False
        self.make_agent()

    def make_agent(self):
        self.obs_in = tf.placeholder(tf.float32, shape=[None, self.obs_size])
        self.Qs_gt = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.act_num = tf.placeholder(tf.int32, shape=[None])

        Q_mask = tf.one_hot(self.act_num, self.action_size)

        # vars
        W_fc1 = tf.Variable(tf.truncated_normal([self.obs_size, 32], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[32]))

        W_fc2 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[32]))

        W_fc3 = tf.Variable(tf.truncated_normal([32, self.action_size], stddev=0.1))
        b_fc3 = tf.Variable(tf.constant(0.1, shape=[self.action_size]))

        h_fc1 = tf.nn.relu(tf.matmul(self.obs_in, W_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        self.Qs_nn = tf.matmul(h_fc2, W_fc3) + b_fc3

        self.Q_masked = tf.multiply(self.Qs_nn, Q_mask)

        self.loss = tf.reduce_mean(
                        tf.square(tf.subtract(self.Q_masked, self.Qs_gt)))

        self.train_step = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord = self.coord)

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

            time.sleep(0.05)
        return total_steps

    def agent_action(self, observation):
        if self.agent_is_trained or True:
            predicted_Qs = self.sess.run(self.Qs_nn, feed_dict={self.obs_in:[observation]})
            action = np.argmax(predicted_Qs)
        else:
            action = self.env.action_space.sample()
        return action

    def train_agent(self):
        # print "Training Agent"
        batch_memory = random.sample(self.replay_memory, self.SAMPLE_TRAIN_SIZE)

        max_predicted_Qs = np.zeros(self.SAMPLE_TRAIN_SIZE)
        for i in range(self.SAMPLE_TRAIN_SIZE):
            qs = self.agent_action(batch_memory[i][0])
            max_predicted_Qs[i] = np.max(qs)

        max_target_Q = np.zeros(self.SAMPLE_TRAIN_SIZE)
        target_Qs = np.zeros((self.SAMPLE_TRAIN_SIZE, self.action_size))
        target_actions = np.zeros(self.SAMPLE_TRAIN_SIZE)
        for i in range(self.SAMPLE_TRAIN_SIZE):
            state_obs, action, reward, next_state_obs, is_done = batch_memory[i]

            max_target_Q[i] = reward 
            if not is_done:
                max_target_Q[i] += self.QL_GAMMA * max_predicted_Qs[i]
            target_Qs[i][action] = max_target_Q[i]
            target_actions[i] = action

        features = [f[0] for f in batch_memory]

        # print features[:10]
        # print target_Qs[:10]

        _, loss, Qs_nn, Q_masked = self.sess.run([self.train_step, self.loss, 
                                                    self.Qs_nn, self.Q_masked], 
                    feed_dict={self.obs_in:features, self.Qs_gt:target_Qs,
                                self.act_num:target_actions})

        # print "Agent Score", self.agent.score(features, target_Qs)
        # print self.agent.predict(features)[:10]
        if not self.agent_is_trained:
            self.agent_is_trained = True
            print "Trained"

        return loss, Qs_nn, target_Qs, Q_masked

    def train(self):
        self.initialize()

        agent_steps = []
        overall_steps = 0
        for ep in range(self.MAX_EPISODES):
            if ep % self.WATCH_INTERVAL == 0:
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

                self.replay_memory.append([obs, action, reward, new_obs, is_done])
                if len(self.replay_memory) > self.MAX_REPLAY_STEPS:
                    self.replay_memory.pop(0)

                if is_done:
                    # print("Episode finished after {} timesteps".format(step+1))
                    break

                obs = new_obs
                # time.sleep(0.1)

                if len(self.replay_memory) > self.SAMPLE_TRAIN_SIZE:
                    loss, Qs_nn, target_Qs, Q_masked = self.train_agent()
                    if overall_steps % 100 == 0:
                        print "Episode", ep, "Total Steps", overall_steps, "Loss", loss
                        # for i in range(2):
                        #     print Qs_nn[i], Q_masked[i], target_Qs[i]

                overall_steps += 1

        plt.plot(range(self.MAX_EPISODES/self.WATCH_INTERVAL), agent_steps)
        plt.show()


if __name__ == "__main__":
    with tf.Session() as sess:
        ql = QL_NN('CartPole-v0', sess)
        # ql.random_play()

        ql.train()