#coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as funct
import numpy as np
import gym
import math
from gym import wrappers
from matplotlib import pyplot as plt

global EPSILON
MINI_BATCH = 32
ALPHA = 0.008                  
EPSILON = 1               
GAMMA = 0.99                 
TARGET_UPDATE_FREQUENCY = 100   # q target 更新频率
MEMORY_CAPACITY = 2000

class Net(nn.Module):
    def __init__(self, STATE_NUM, ACTION_NUM, hidden_num):
        super(Net, self).__init__()
        self.incoming = nn.Linear(STATE_NUM, hidden_num)
        self.incoming.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(hidden_num, ACTION_NUM)
        self.out.weight.data.normal_(0, 0.1)   

    def forward(self, x):
        x = self.incoming(x)
        x = funct.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, STATE_NUM, ACTION_NUM, ENV_A_SHAPE, hidden_num):
        self.eval_net, self.target_net = Net(STATE_NUM, ACTION_NUM, hidden_num), Net(STATE_NUM, ACTION_NUM, hidden_num)

        self.learn_count = 0   #用于q target的更新
        self.memery_count = 0        #记录memory中存储了多少条记录了
        self.memory = np.zeros((MEMORY_CAPACITY, STATE_NUM * 2 + 2))     # 初始化memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=ALPHA)
        self.loss_func = nn.MSELoss()
        self.state_num = STATE_NUM
        self.action_num = ACTION_NUM
        self.env_a_shape = ENV_A_SHAPE
        self.loss = 0
        print(self.memory)

    def get_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() > EPSILON:   #选择最大
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy() #取得最大值的位置索引
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.action_num)
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memery_count % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memery_count += 1

    def learn(self):
        # target 参数更新
        if self.learn_count % TARGET_UPDATE_FREQUENCY == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_count += 1

        # Sample random mini-batch of transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, MINI_BATCH)
        b_memory = self.memory[sample_index, :]  #挑选出sample_index对应的行的数据，有MINI_BATCH行数据
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.state_num]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.state_num:self.state_num+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.state_num+1:self.state_num+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.state_num:]))

        # 梯度下降法更新theata
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(MINI_BATCH, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.loss += float(loss.data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_loss(self):
        result=self.loss
        self.loss=0
        return result

def cartpole():
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    global EPSILON

    EPSILON = 1
    ACTION_NUM = env.action_space.n
    STATE_NUM = env.observation_space.shape[0]
    #print (STATE_NUM)
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
    NUM_EPISODES = 200
    MAX_T = 20000
    STREAK_TO_END = 50
    FINAL_T = 199
    streaks = 0
    hidden_num = 50
    dqn = DQN(STATE_NUM, ACTION_NUM, ENV_A_SHAPE, hidden_num)

    #train
    train_loss = []
    result = []
    for i_episode in range(NUM_EPISODES):
        s = env.reset()
        reward = 0
        train_t = []
        for t in range(MAX_T):
            #env.render()
            a = dqn.get_action(s)
            s_dot, r, done, info = env.step(a)

            reward += r
            # 修改reward值
            x, x_dot, theta, theta_dot = s_dot
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_dot)

            
            #如果memory满了就开始学习
            if dqn.memery_count > MEMORY_CAPACITY:
                #env.render()
                dqn.learn()
                if done:
                    #result.append(reward)
                    print('Episode: ', i_episode,' reward: ', reward, ' t: ', t)
                    break
            if (t >= FINAL_T):
                streaks += 1
            else:
                streaks = 0
            if t==MAX_T - 1:
                result.append(reward)
            if done:
                result.append(reward)
                break
            train_t.append(t)
            s = s_dot
        train_loss.append(dqn.get_loss()/train_t[-1])
        if streaks > STREAK_TO_END:
            print("cartpole finished streaks:", streaks)
            break
        EPSILON = max(0.01, min(1, 1.0 - math.log10((i_episode+1)/25.0)))
    
    plt.plot(train_loss)
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.show()

    plt.plot(result)
    plt.xlabel("round")
    plt.ylabel("reward")
    plt.show()

    print('test start!')
    #test
    env = env.unwrapped
    env = wrappers.Monitor(env, '/tmp/cartpole-3', force=True)
    result = []
    ts = []
    for i_episode in range(NUM_EPISODES):
        s = env.reset()
        reward = 0
        
        for t in range(MAX_T):
            #env.render()
            a = dqn.get_action(s)
            s_dot, r, done, info = env.step(a)

            reward += r

            if t == MAX_T - 1:
                ts.append(t)
            if done:
                print(t)
                ts.append(t)
                
                break
            s = s_dot
        result.append(reward)
        print('episode:', i_episode)
    print("mean steps:", np.mean(ts))
    print("mean reward:", np.mean(result))
    print("standard deviation:", np.std(result))

def mountaincar():
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    print(env.observation_space.high[0])
    global EPSILON
    EPSILON = 1
    ACTION_NUM = env.action_space.n
    STATE_NUM = env.observation_space.shape[0]
    #print (STATE_NUM)
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
    NUM_EPISODES = 400
    MAX_T = 2000
    STREAK_TO_END = 300
    hidden_num = 25
    dqn = DQN(STATE_NUM, ACTION_NUM, ENV_A_SHAPE, hidden_num)

    #train
    train_loss = []
    result = []
    streaks = 0
    for i_episode in range(NUM_EPISODES):
        s = env.reset()
        reward = 0
        end_flag = False

        train_t = []
        for t in range(MAX_T):
            #env.render()
            a = dqn.get_action(s)
            s_dot, r, done, info = env.step(a)
            reward += r

            # 修改reward值
            x, x_dot = s_dot
            if x > 0.5:
                r1 = (x - (-0.5)) * 120
            # elif x > 0.35:
            #     r1 = (x - (-0.5)) * 40
            else:
                r1 = (abs(x - (-0.5))) - 0.2
            r2 = abs(x_dot) * 5
            r = r1 

            dqn.store_transition(s, a, r, s_dot)

            
            if dqn.memery_count > MEMORY_CAPACITY:
                #env.render()
                dqn.learn()
                if done:
                    #result.append(reward)
                    print('Episode: ', i_episode,' reward: ', reward, 't:', t)

            if t == MAX_T - 1:
                end_flag = False
            if done:
                end_flag = True
                streaks += 1
                break
            s = s_dot
            train_t.append(t)
        result.append(reward)
        train_loss.append(dqn.get_loss()/ train_t[-1])
        if end_flag == False:
            streaks = 0
        if streaks > STREAK_TO_END:
            print('mountaincar finished')
            break
        print('one episode finished!', i_episode)
        EPSILON = max(0.01, min(1, 1.0 - math.log10((i_episode+1)/25.0)))
    
    plt.plot(train_loss)
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.show()

    plt.plot(result)
    plt.xlabel("round")
    plt.ylabel("reward")
    plt.show()

    #test
    env = env.unwrapped
    result = []
    for i_episode in range(NUM_EPISODES):
        s = env.reset()
        reward = 0
        
        for t in range(MAX_T):
            #env.render()
            a = dqn.get_action(s)
            s_dot, r, done, info = env.step(a)

            reward += r

            if done:
                print(t)
                result.append(reward)
                break
            s = s_dot
    print("mean reward:", np.mean(result))
    print("standard deviation:", np.std(result))

def acrobot():
    env = gym.make('Acrobot-v1')
    # env = env.unwrapped
    print(env.observation_space.high[0])
    global EPSILON
    EPSILON = 1
    ACTION_NUM = env.action_space.n
    STATE_NUM = env.observation_space.shape[0]
    #print (STATE_NUM)
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
    NUM_EPISODES = 300
    MAX_T = 2000
    STREAK_TO_END = 200
    hidden_num = 75
    dqn = DQN(STATE_NUM, ACTION_NUM, ENV_A_SHAPE, hidden_num)

    #train
    train_loss =[]
    result = []
    streaks = 0
    for i_episode in range(NUM_EPISODES):
        s = env.reset()
        reward = 0
        end_flag = False
        train_t = []
        for t in range(MAX_T):
            #env.render()
            a = dqn.get_action(s)
            s_dot, r, done, info = env.step(a)

            # 修改reward值
            # x, x_dot = s_dot
            # if x > 0.5:
            #     r1 = (x - (-0.5)) * 90
            # else:
            #     r1 = (abs(x - (-0.5))) - 0.2
            # r2 = abs(x_dot) * 5
            # r = r1 

            dqn.store_transition(s, a, r, s_dot)

            reward += r
            if dqn.memery_count > MEMORY_CAPACITY:
                #env.render()
                dqn.learn()
                if done:
                    #result.append(reward)
                    print('Episode: ', i_episode,' reward: ', reward, 't:', t)

            if done:
                end_flag = False
                streaks += 1
                break
            s = s_dot
            train_t.append(t)
        result.append(reward)
        train_loss.append(dqn.get_loss()/ train_t[-1])
        if end_flag == False:
            streaks = 0
        if streaks > STREAK_TO_END:
            print('Acrobot finished')
            break
        print('one episode finished!')
        EPSILON = max(0.01, min(1, 1.0 - math.log10((i_episode+1)/25.0)))
    
    plt.plot(train_loss)
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.show()

    plt.plot(result)
    plt.xlabel("round")
    plt.ylabel("reward")
    plt.show()
    
    #test
    env = env.unwrapped
    result = []
    for i_episode in range(NUM_EPISODES):
        s = env.reset()
        reward = 0
        
        for t in range(MAX_T):
            #env.render()
            a = dqn.get_action(s)
            s_dot, r, done, info = env.step(a)

            reward += r

            if done:
                print(t)
                result.append(reward)
                break
            s = s_dot
    print("mean reward:", np.mean(result))
    print("standard deviation:", np.std(result))

if __name__ == '__main__':
    cartpole()
    # mountaincar()
    # acrobot()