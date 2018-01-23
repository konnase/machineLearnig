#coding=utf-8
#有超过100次实验中，小车至少移动了195步而杆子不倒，才算解决了cartpole问题。
import gym
import numpy as np
import random
import math
from gym import wrappers

def get_state_threshold(STATE_INDEX, env):
    state_threshold = []
    for i in range(STATE_INDEX):
        #print (env.observation_space.low)
        state_threshold.append([env.observation_space.low[i], env.observation_space.high[i]])
    #print (state_threshold)
    return state_threshold


global EPSILON
global ALPHA
global GAMMA
global Q

def train_cartpole():
    env = gym.make('CartPole-v0')
    #env = env.unwrapped

    # 每个状态维度的状态离散数量
    STATE_NUM = (1, 1, 6, 5)  # (x, x', theta, theta')

    STATE_INDEX = len(STATE_NUM)
    # 每个离散状态的边界值
    STATE_THRESHOLD = get_state_threshold(STATE_INDEX, env)
    STATE_THRESHOLD[1] = [-0.5, 0.5]
    STATE_THRESHOLD[3] = [-1, 1]

    global EPSILON
    global ALPHA
    global GAMMA 
    global Q
    EPSILON = 1
    ALPHA = 0.5
    GAMMA = 0.99 
    streaks = 0
    ## 为每个状态动作对创建Q表
    Q = np.zeros(STATE_NUM + (env.action_space.n,))

    NUM_EPISODES = 1000
    MAX_T = 20000
    STREAK_TO_END = 200
    FINAL_T = 199
    #train
    for episode in range(NUM_EPISODES):
        observation = env.reset()
        state = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)

        #Q-Learning algorithm
        for t in range(MAX_T):
            #env.render()
            action = get_action(state, env)
            observation, reward, done, _ = env.step(action)
            #新状态
            state_dot = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
            best_q = np.amax(Q[state_dot])
            Q[state + (action,)] += ALPHA*(reward + GAMMA*(best_q) - Q[state + (action,)])
            state = state_dot

            if done:
               print("Episode %d ; t: %d" % (episode, t))
               if (t >= FINAL_T):
                   streaks += 1
               else:
                   streaks = 0
               break

            #sleep(0.25)

        # It's considered done when it's solved over 120 times consecutively
        print (t)
        if streaks > STREAK_TO_END:
            print("cartpole finished, streaks: ", streaks)
            break

        # Update parameters
        #print (math.log10(1/25))
        EPSILON = max(0.01, min(1, 1.0 - math.log10((episode+1)/25.0)))
        ALPHA = max(0.1, min(0.5, 1.0 - math.log10((episode+1)/25.0)))
        # EPSILON = max(0.01, 2 - 2/(1 + math.exp((-episode/30))))
        # ALPHA = max(0.1, min(0.5, 1/(1 + math.exp((episode - 30)/80))))
    
    #test
    result=[]
    rewards = [] #记录每条轨迹的reward之和
    #env = wrappers.Monitor(directory='/tmp/CartPole-v0',video_callable=False, write_upon_reset=True)(env)
    env = env.unwrapped
    env = wrappers.Monitor(env, '/tmp/cartpole', force=True)
    
    #env.monitor.start('/tmp/cartpole-experiment-1',video_callable=lambda count: count % 10 == 0, force=True)
    for i in range(NUM_EPISODES):
        observation=env.reset()
        state = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
        t=0
        sum_reward = 0
        while (t< MAX_T):
            env.render()
            t+=1
            action = np.argmax(Q[state])
            observation, reward, done, info = env.step(action)
            sum_reward += reward
            #print (reward)
            new_state=observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
            state=new_state
            if done:
                result.append(t)
                print(t)
                break
        rewards.append(sum_reward)
        sum_reward = 0
    print("mean time steps:",np.mean(result))
    print("mean rewards:", np.mean(rewards))
    print("standard deviation:",np.std(rewards))
    #env.monitor.close()



def train_mountaincar():
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    # 每个状态维度的状态离散数量
    STATE_NUM = (8, 8)  # (x, x', theta, theta')

    STATE_INDEX = len(STATE_NUM)
    # 每个离散状态的边界值
    STATE_THRESHOLD = get_state_threshold(STATE_INDEX, env)

    global EPSILON
    global ALPHA
    global GAMMA 
    global Q
    EPSILON = 1
    ALPHA = 0.5
    GAMMA = 0.99 
    streaks = 0
    ## 为每个状态动作对创建Q表
    Q = np.zeros(STATE_NUM + (env.action_space.n,))

    NUM_EPISODES = 1000
    MAX_T = 2000
    STREAK_TO_END = 200
    FINAL_T = 199
    #train
    for episode in range(NUM_EPISODES):
        observation = env.reset()
        state = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
        end_flag = False
        #Q-Learning algorithm
        for t in range(MAX_T):
            #env.render()
            action = get_action(state, env)
            observation, reward, done, _ = env.step(action)
            #新状态
            state_dot = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
            #返回最大值
            best_q = np.amax(Q[state_dot])
            Q[state + (action,)] += ALPHA*(reward + GAMMA*(best_q) - Q[state + (action,)])
            state = state_dot

            if done:
                print("Episode %d ; t: %d" % (episode, t))
                end_flag = True
                streaks += 1
                break

            #sleep(0.25)
        #如果在一个episode中（即MAX_T步后），小车不能达到顶点，
        if end_flag == False:
            streaks = 0
        print (t)
        if streaks > STREAK_TO_END:
            print("mountaicar finished")
            break

        # Update parameters
        EPSILON = max(0.01, min(1, 1.0 - math.log10((episode+1)/25.0)))
        ALPHA = max(0.1, min(0.5, 1.0 - math.log10((episode+1)/25.0)))
        # EPSILON = max(0.01, 2 - 2/(1 + math.exp((-episode/30))))
        # ALPHA = max(0.1, min(0.5, 1/(1 + math.exp((episode - 30)/80))))

    #test
    result=[]
    rewards = [] #记录每条轨迹的reward之和
    env = env.unwrapped
    #env.monitor.start('/tmp/mountaincar-experiment-1',video_callable=lambda count: count % 10 == 0)
    for i in range(NUM_EPISODES):
        observation=env.reset()
        state = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
        t=0
        sum_reward = 0
        while (t< MAX_T):
            #env.render()
            t+=1
            action = np.argmax(Q[state])
            observation, reward, done, info = env.step(action)
            sum_reward += reward
            #print (reward)
            new_state=observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
            state=new_state
            if done:
                result.append(t)
                print(t)
                break
        rewards.append(sum_reward)
        sum_reward = 0
    print("mean time steps:",np.mean(result))
    print("mean rewards:", np.mean(rewards))
    print("standard deviation:",np.std(rewards))
    #env.monitor.close()



def train_acrobot():
    env = gym.make('Acrobot-v1')
    env = env.unwrapped

    # 每个状态维度的状态离散数量
    STATE_NUM = (1, 1, 1, 1, 10, 10)  #[cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

    STATE_INDEX = len(STATE_NUM)
    # 每个离散状态的边界值
    STATE_THRESHOLD = get_state_threshold(STATE_INDEX, env)

    global EPSILON
    global ALPHA
    global GAMMA 
    global Q
    EPSILON = 1
    ALPHA = 0.5
    GAMMA = 0.99 
    streaks = 0
    ## 为每个状态动作对创建Q表
    Q = np.zeros(STATE_NUM + (env.action_space.n,))
    print(env.action_space.n)
    NUM_EPISODES = 1000
    MAX_T = 2000
    STREAK_TO_END = 200
    FINAL_T = 199
    for episode in range(NUM_EPISODES):
        observation = env.reset()
        state = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
        end_flag = False
        #Q-Learning algorithm
        for t in range(MAX_T):
            #env.render()
            action = get_action(state, env)
            observation, reward, done, _ = env.step(action)
            # position, velocity = observation
            # reward = abs(position - (-0.5))
            #新状态
            state_dot = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
            #返回最大值
            best_q = np.amax(Q[state_dot])
            Q[state + (action,)] += ALPHA*(reward + GAMMA*(best_q) - Q[state + (action,)])
            state = state_dot

            if done:
                print("Episode %d ; t: %d" % (episode, t))
                end_flag = True
                streaks += 1
                break

            #sleep(0.25)
        #如果在一个episode中（即MAX_T步后），小车不能达到顶点，
        if end_flag == False:
            streaks = 0
        print (t)
        if streaks > STREAK_TO_END:
            print("acrobot finished")
            break

        # Update parameters
        EPSILON = max(0.01, min(1, 1.0 - math.log10((episode+1)/25.0)))
        ALPHA = max(0.1, min(0.5, 1.0 - math.log10((episode+1)/25.0)))
        # EPSILON = max(0.01, 2 - 2/(1 + math.exp((-episode/30))))
        # ALPHA = max(0.1, min(0.5, 1/(1 + math.exp((episode - 30)/80))))

    #test
    result=[]
    rewards = [] #记录每条轨迹的reward之和
    
    #env = wrappers.Monitor( env, '/tmp/acrobot-experiment-1',video_callable=lambda count: count % 10 == 0)
    env = env.unwrapped
    for i in range(NUM_EPISODES):
        observation=env.reset()
        state = observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
        t=0
        sum_reward = 0
        while (t< MAX_T):
            #env.render()
            t+=1
            action = np.argmax(Q[state])
            observation, reward, done, info = env.step(action)
            sum_reward += reward
            #print (reward)
            new_state=observation_to_state(observation, STATE_THRESHOLD, STATE_NUM)
            state=new_state
            if done:
                result.append(t)
                print(t)
                break
        rewards.append(sum_reward)
        sum_reward = 0
    print("mean time steps:",np.mean(result))
    print("mean rewards:", np.mean(rewards))
    print("standard deviation:",np.std(rewards))
    #env.monitor.close()

def get_action(state, env):
    p = np.random.uniform(0,1)
    if p < EPSILON:
        action = env.action_space.sample()
    else:
        #返回最大值所在的下标
        action = np.argmax(Q[state])
    return action

def observation_to_state(observation, STATE_THRESHOLD, STATE_NUM):
    state = []
    for i in range(len(observation)):
        # print(i)
        # print(observation)
        if observation[i] <= STATE_THRESHOLD[i][0]:
            index = 0
        elif observation[i] >= STATE_THRESHOLD[i][1]:
            index = STATE_NUM[i] - 1
        else:
            # 将观测值离散化为状态值
            bound_width = STATE_THRESHOLD[i][1] - STATE_THRESHOLD[i][0]
            observation_scaling = (observation[i] - STATE_THRESHOLD[i][0])/bound_width
            index = int(round(observation_scaling * (STATE_NUM[i] - 1)))
        state.append(index)
    return tuple(state)

if __name__ == "__main__":
    train_cartpole()
    # train_mountaincar()
    # train_acrobot()
    