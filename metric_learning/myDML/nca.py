import numpy as np


class nca:
    def __init__(self,traindata):
        self.train_Data = traindata[0]
        self.train_Lable = traindata[1]
        #变换矩阵
        self.A = np.eye(self.train_Data.shape[1])
        #记录所有点之间的距离
        self.distance = np.zeros([len(self.train_Lable),len(self.train_Lable)])
        #记录每个点到其他点的距离之和
        self.distanceI = np.zeros(len(self.train_Lable))

        print("----------------")
        print(self.distance[0][1])


    def distance_between_i_and_j(self):
        l = len(self.train_Lable)

        for i in range(l):
            ax_i = np.dot(self.A, np.transpose([self.train_Data[i]])) #dX1列向量ax_i
            for j in range(i+1):
                ax_j = np.dot(self.A,np.transpose([self.train_Data[j]]))
                self.distance[i][j] = np.exp(-np.sum(np.square(ax_i-ax_j)))
                self.distance[j][i] = self.distance[i][j]
                #print(self.distance[i][j])
        for i in range(l):
            # for j in range(l):
            #     if i!=j:
            #         self.distanceI[i]+=self.distance[i][j]
            self.distanceI[i] = np.sum(self.distance[i]) - self.distance[i][i]
            #print(self.distanceI[i])

    #i选择j并继承其标签的概率
    def p_ij(self,i,j):
        if i == j:
            return 0
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

    def derivative_a(self,i):
        l = len(self.train_Lable)
        #deriv_a = np.zeros(self.A.shape,self.A.shape)
        # for i in range(l):
        sum_k = np.zeros(self.A.shape)

        sum_j = np.zeros(self.A.shape)
        for k in range(l):
            sum_k = sum_k + self.p_ij(i,k) * np.dot(np.transpose([self.x_ij(i,k)]),[self.x_ij(i,k)] )
        for j in range(l):
            if self.train_Lable[j]==self.train_Lable[i]:
                sum_j = sum_j + self.p_ij(i,j) * np.dot(np.transpose([self.x_ij(i,j)]),[self.x_ij(i,j)])
        deriv_a = self.p_i(i) * sum_k - sum_j
        return 2 * np.dot(self.A , deriv_a)

    def train(self,count=50,rate=0.02):
        for i in range(count):
            print("%d %d", i , i%len(self.train_Lable))
            self.distance_between_i_and_j()
            #print(self.A.shape)
            self.A += rate * self.derivative_a(i%len(self.train_Lable))
    def get_m(self):
        #print(np.dot(self.A.T,self.A))
        return np.dot(self.A.T,self.A)

