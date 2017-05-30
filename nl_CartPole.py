import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print "Obs", observation
        action = env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        print "Action", action, "Reward", reward, "Done", done, "Info", info
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

        observation = new_observation

# reference https://gist.github.com/viswanathgs/ca9788020cfcc7849b9181d9239a2ef4