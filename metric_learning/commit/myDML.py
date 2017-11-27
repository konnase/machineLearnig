# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
import time
from threading import Thread
import functools
from randnca import randnca

# (global) variable definition here
TRAINING_TIME_LIMIT = 60*10

global ncas
# class definition here

# function definition here
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    global ncas
    ncas = randnca(traindata)
    ncas.train(10000,0.006)

    # 在此处完成你的训练函数，注意训练时间不要超过TRAINING_TIME_LIMIT(秒)。
    #time.sleep(1) # 这行仅用于测试训练超时，运行时请删除这行，否则你的TRAINING_TIME_LIMIT将-1s。
    return 0

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):
    d = inst_a - inst_b
    #print(ncas.get_m())
    dist = np.dot(np.dot(d,ncas.get_m()),np.transpose([d]))
    return dist

# main program here
if  __name__ == '__main__':
    pass
