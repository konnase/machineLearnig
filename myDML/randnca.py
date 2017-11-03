import numpy as np
import random


class randnca:
    def __init__(self,traindata):
        self.train_Data = traindata[0]
        self.train_Lable = traindata[1]
        #变换矩阵
        self.A = np.eye(self.train_Data.shape[1])
        #记录所有点之间的距离
        self.distance = np.eye(len(self.train_Lable))
        #记录每个点到其他点的距离之和
        self.distanceI = np.ones(len(self.train_Lable))

        print("----------------")
        print(len(self.train_Data))
        print(self.distanceI)

    def distance_between_i_and_j(self,i):
        l = len(self.train_Data)

        #for i in range(l):

        ax_i = np.dot(self.A, np.transpose([self.train_Data[i]])) #dX1列向量ax_i
        for j in range(l):
            ax_j = np.dot(self.A,np.transpose([self.train_Data[j]]))
            self.distance[i][j] = np.exp(-np.sum(np.square(ax_i-ax_j)))
            self.distance[j][i] = self.distance[i][j]
            #print(self.distance[i][j])
        #for i in range(l):
        for j in range(l):
            if i!=j:
                self.distanceI[i]+=self.distance[i][j]
        # for p in range(l):
        #     print(self.distance[i][p])
        #self.distanceI[i] = np.sum(self.distance[i]) - self.distance[i][i]
        print("距离之和：%f" % self.distanceI[i])

    #i选择j并继承其标签的概率
    def p_ij(self,i,j):
        if i == j:
            return 0
        #print("divide %d" , i)
        return self.distance[i][j]/self.distanceI[i]

    #i被正确分类的概率
    def p_i(self,i):
        pi = 0
        for j in range(len(self.train_Lable)):
            if self.train_Lable[j]==self.train_Lable[i]:
                pi = pi + self.p_ij(i,j)
        return pi

    def x_ij(self,i,j):
        return self.train_Data[i]-self.train_Data[j]

    def derivative_a(self,iter,rand):
        l = len(self.train_Lable)
        #deriv_a = np.zeros(self.A.shape)
        #for i in range(l):
        i = rand
        sum_k = np.zeros(self.A.shape)

        sum_j = np.zeros(self.A.shape)
        for k in range(l):
            #print("derivate_a p_ij %d" , i)
            sum_k = sum_k + self.p_ij(i,k) * np.dot(np.transpose([self.x_ij(i,k)]),[self.x_ij(i,k)] )
        for j in range(l):
            if self.train_Lable[j]==self.train_Lable[i]:
                sum_j = sum_j + self.p_ij(i,j) * np.dot(np.transpose([self.x_ij(i,j)]),[self.x_ij(i,j)])

        #print("derivate_a pi divide %d" , i)
        deriv_a = self.p_i(i) * sum_k - sum_j
        return 2 * np.dot(self.A , deriv_a)

    def train(self,count=1000,rate=0.02):
        for i in range(count):
            print("%d %d", i , i%len(self.train_Lable))
            rand = random.randint(0,len(self.train_Lable)-1)
            #随机取一个样本进行度量
            self.distance_between_i_and_j(rand)
            print(self.A.shape,rand)
            self.A += rate * self.derivative_a(i%len(self.train_Lable),rand)
    def get_m(self):
        #print(np.dot(self.A.T,self.A))
        return np.dot(self.A.T,self.A)

