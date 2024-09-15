import gym
import turtle
import numpy as np

class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)


class SarsaAgent:
    def __init__(self, env,cfg):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.Q = np.zeros([self.n_states, self.n_actions])
        self.epsilon = cfg.epsilon_max
        self.epsilon_max = cfg.epsilon_max
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.sample_count = 0
        self.name = "SarsaAgent"

    def choose_action(self, state): # 训练用
        self.sample_count += 1
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -self.sample_count / self.epsilon_decay)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state, :])
        
    def predict(self, state): # 测试用
        return np.argmax(self.Q[state, :])
    
    def sarsa(self, state, action, reward, next_state,next_action,done):
        if done:
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
        else:
            self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])


class QLearningAgent:
    def __init__(self, env,cfg):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.Q = np.zeros([self.n_states, self.n_actions])
        self.epsilon = cfg.epsilon_max
        self.epsilon_max = cfg.epsilon_max
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.sample_count = 0
        self.name = "QLearningAgent"

    def choose_action(self, state): # 训练用
        self.sample_count += 1
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -self.sample_count / self.epsilon_decay)
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state, :])
        
    def predict(self, state): # 测试用
        return np.argmax(self.Q[state, :])

    def qlearning(self, state, action, reward, next_state, done):
        if done:
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
        else:
            self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])

class Config:
    def __init__(self):
        self.seed = 1 # 随机种子
        self.epsilon = 0.95 #  e-greedy策略中epsilon的初始值
        self.epsilon_max = 0.95 #  e-greedy策略中epsilon的初始值
        self.epsilon_min = 0.01 #  e-greedy策略中epsilon的最终值
        self.epsilon_decay = 200 #  e-greedy策略中epsilon的衰减率
        self.gamma = 0.9 # 折扣因子
        self.alpha = 0.1 # 学习率

def train(env,agent,max_episodes):
    print(">>>>> Training started with {}".format(agent.name))
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        rewards = []
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            next_action = agent.choose_action(next_state)
            if agent.name == "QLearningAgent":
                agent.qlearning(state, action, reward, next_state, done)
            if agent.name == "SarsaAgent":
                agent.sarsa(state, action, reward, next_state,next_action,done)
            state = next_state
            if done:
                break
        rewards.append(episode_reward)
        if episode % 100 == 0:
            print("Episode:{} Reward:{:.2f} Epsilon:{}".format(episode, episode_reward,agent.epsilon))
    print(">>>>> Training finished\n")
    return rewards

def test(env,agent): # 测试
    print(">>>>> Testing started with {}".format(agent.name))
    state = env.reset()
    episode_reward = 0
    while True:
        action = agent.predict(state)
        action_lang=["👆","👉","👇","👈"]
        print("state:{} action:{}".format(state,action_lang[action]))
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            print("Reward:{:.2f}".format(episode_reward))
            break
        state = next_state
    print(">>>>> Testing finished\n")

def smooth(data, weight=0.9):  
    '''用于平滑曲线
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

import matplotlib.pyplot as plt
def plot_rewards(rewards,title="learning curve"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlim(0, len(rewards), 10)  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()

cfg = Config()
env_name = 'CliffWalking-v0'
env = gym.make(env_name)
env = CliffWalkingWapper(env)

sarsa_agent = SarsaAgent(env,cfg)
qlearning_agent = QLearningAgent(env,cfg)

ql_rewards = train(env,qlearning_agent,600)
# plot_rewards(ql_rewards, title=f"training curve on {qlearning_agent.name}")  

sa_rewards = train(env,sarsa_agent,600)
# plot_rewards(sa_rewards, title=f"training curve on {sarsa_agent.name}")  

test(env,qlearning_agent)
test(env,sarsa_agent)

