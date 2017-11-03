import numpy as np
import operator
import pickle as pkl
import nca
import myDML

from numpy.ma import array

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # 数据集大小
    # 计算距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  #计算各个样本点到测试样本点的向量
    sqDiffMat = diffMat**2  #平方
    sqDistances = sqDiffMat.sum(axis=1)  #矩阵按行相加
    distances = sqDistances**0.5   #开方
    # 按距离排序
    sortedDistIndicies = distances.argsort()   #返回升序排序后各个值对应的下标
    # 统计前k个点所属的类别
    classCount = {}
    for i in range(k):
        votaIlabel = labels[sortedDistIndicies[i]]
        classCount[votaIlabel] = classCount.get(votaIlabel, 0) + 1   #在字典类型classCount中取label出现的次数，如果没有则返回默认值0,否则加一
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print((sortedClassCount))
    # 返回前k个点中频率最高的类别
    return sortedClassCount[0][0]

if __name__ == '__main__':
    train_file = open('dataset/moons/moons_train.pkl', 'rb')
    test_file = open('dataset/moons/moons_test.pkl', 'rb')
    moons_train = pkl.load(train_file)  # tuple of training data
    moons_test = pkl.load(test_file)  # tuple of testing data
    train_file.close()
    test_file.close()

    train_X = moons_train[0]  # instances of training data
    train_Y = moons_train[1]  # labels of training data
    test_X = moons_test[0]  # instances of testing data
    test_Y = moons_test[1]  # labels of testing data
    # group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # labels = ['A', 'A', 'B', 'B']
    # a = classify([0,0],group,labels,3)
    # print(a)
    # print(zeros([3,3]))

    a = np.ones(4)
    print(a)
    print("---------------------")
    print(np.transpose([a]))
    # moons_train = array([array([[0,3],
    #                            [4,0],
    #                            [0,1],
    #                            [1,0],
    #                            [0,0],
    #                            [1,1],
    #                            [3,0],
    #                            [3,1]]),
    #                      array(['a',
    #                            'b',
    #                            'a',
    #                            'a',
    #                            'a',
    #                            'a',
    #                            'b',
    #                            'b'])])
    #myDML.train(moons_train)

