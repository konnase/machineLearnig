import numpy as np
import sys

if sys.version_info.major == 3:
    xrange = range

# five elements for HMM
states = ('Healthy', 'Fever')

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}

emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}
# Helps visualize the steps of Viterbi.
def print_dptable(V):
    print("    ")
    for i in range(len(V)): print("%7d" % i)
    print

    for y in V[0].keys():
        print("%s: " % y)
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]))
        print

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max([(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    print_dptable(V)
    print(type(path))
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])

def HMMViterbi( a, b, o, pi):
    # Implements HMM Viterbi algorithm

    N = np.shape(b)[0]  # 状态数，值为2
    T = np.shape(o)[0]  # 实际观测序列长度

    path = np.zeros(T)
    delta = np.zeros((N, T))  # 存储生成观测序列的各个状态的概率，每一列为一个观测点上看到的各个状态的概率
    phi = np.zeros((N, T))

    """
    TODO: implement the viterbi algorithm and return path
    """
    paths = {}
    states = ('Healthy', 'Fever')
    for x in states:
        delta[0][x] = pi[x] * b[x][o[0]]  # 初始状态为0的概率乘上在0状态下获得o[0]观测值的概率
        paths[x] = [x]
    for t in xrange(1, T):
        newpath = {}
        for x in states:
            (delta[t][x], state) = max([(delta[t - 1][x1] * a[x1][x] * b[x][o[t]], x1) for x1 in states])
            newpath[state] = paths[state] + [x]
        paths = newpath
    (p, state) = max(([delta[len(o) - 1][x], x]) for x in states)
    for x in paths[state]:
        path.__add__(x)
    return path

if __name__ == '__main__':
    obs = ['normal', 'cold', 'dizzy']
    print(viterbi(obs, states, start_probability, transition_probability, emission_probability))
    #print(HMMViterbi(transition_probability, emission_probability, obs, start_probability))
    # a = [1,2,3,4,5]
    # b = [5,4,3,2,1]
    # print(np.transpose(b))
    # print(a * np.transpose(b))
    print([i for i in range(6, -1, -1)])