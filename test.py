import gym

env = gym.make('CartPole-v0')
env.reset()

for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    env.step(action)
    env.step(action)
env.close