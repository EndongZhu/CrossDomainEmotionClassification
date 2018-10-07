# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:16:30 2016

@author: lenovo
"""

import numpy


def CostFunction(theta , X , Y_ , lam, dim):
    Y = numpy.argmax(Y_,1)
    size = Y.shape[0]
    grad = numpy.zeros(theta.shape) 
    temp = numpy.exp(numpy.dot(X,theta.transpose()))    #先计算出权重和特征的乘积
    neg_like = numpy.zeros(size)
    for i in range(size):
        for j in range(dim):
            if(j == Y[i]):
                neg_like[i] = numpy.log(temp[i][j] / temp[i].sum()) 
    reg = numpy.sum(theta*theta)                        #正则化项
    J = numpy.sum(neg_like)/(-size) + lam*reg/(2*size) #计算出代价函数值
    for j in range(dim):
        var = numpy.zeros([1,grad.shape[1]])
        for i in range(size):
            if(Y[i] == j):
                var = var + X[i]*(1-(temp[i][j]/temp[i].sum()))
            else:
                var = var + X[i]*(-(temp[i][j]/temp[i].sum()))
        grad[j] = var/(-size) + lam*theta[j]/size   # 计算出每一个特征所对应的梯度值（含正则化项）
    return J,grad   # 返回代价函数值和梯度值
    
def corr_cal(x,y):
    size = x.shape[0]
    sum_ = 0
    cnt_ = 0
    for i in range(size):
        linex = x[i,:]
        liney = y[i,:]
        num = numpy.corrcoef(linex,liney)[0,1]
        if not numpy.isnan(num):
            sum_ = sum_ + num
            cnt_ = cnt_ + 1
    return sum_/cnt_
    
def Lcorr_cal(x,y):
    size = x.shape[1]
    LAP = []
    for i in range(size):
        linex = x[:,i]
        liney = y[:,i]
        num = numpy.corrcoef(linex,liney)[0,1]
        LAP.append(num)
    return LAP

def predict(theta,X,Y,dim):	#根据权重进行预测
    size = Y.shape[0]
    temp = numpy.dot(X,theta.transpose())
    neg_like = numpy.zeros(Y.shape)
    for i in range(neg_like.shape[0]):
        for j in range(neg_like.shape[1]):
            neg_like[i][j] = numpy.exp(temp[i][j]) / numpy.sum(numpy.exp(temp[i]))
    AP = corr_cal(neg_like,Y)    
    LAP = Lcorr_cal(neg_like,Y)
    Y_p = numpy.array(numpy.argmax(neg_like,1))
    Y = numpy.array(numpy.argmax(Y,1))
    cnt = 0    
    for i in range(size):
        if(Y_p[i] == Y[i]):
            cnt = cnt+1
    return AP,LAP,cnt/size
    

a = numpy.loadtxt("..\whole",delimiter=",")
a = numpy.transpose(numpy.transpose(a[:,1:]))
b = numpy.loadtxt("..\label_train",delimiter=",")
b = numpy.transpose(numpy.transpose(b[:,1:]))
c = numpy.loadtxt("..\label_test",delimiter=",")
c = numpy.transpose(numpy.transpose(c[:,1:]))

train_num = 246
test_num = 1000

X_source = a[:train_num]
Y_source = b
X_target = a[train_num:] 
Y_target = c


dim = 6
word_num = a.shape[1]
step = 10
iter_times = 1000
lam = 0.5

step = 10

output = open('res_' , 'w' , encoding= 'utf8')

for num_ratio in [1/8]:
    num = round((a.shape[0]-train_num)*num_ratio)
    
    X_training = X_target[:num]
    Y_training = Y_target[:num]
    X_testing = X_target[num:]
    Y_testing = Y_target[num:]
    
    feature_num = X_training.shape[1]
    theta = numpy.zeros([dim,feature_num])
    for i in range(iter_times):
        J,grad = CostFunction(theta , X_training , Y_training , lam , dim)
        theta = theta - step*grad
    AP1,LAP1,pred1 = predict(theta,X_training,Y_training,dim)
    AP2,LAP2,pred2 = predict(theta,X_testing,Y_testing,dim)
    
    print("Accuracy#%f: %f\n"%(num_ratio, pred1))
    print("AP#%f: %f\n"%(num_ratio, AP2))
    print("LAP#%f: "%(num_ratio))
    for num in LAP2:
        print("%f "%(num))
    print("\n")
    
    output.write("Accuracy#%f: %f\n"%(num_ratio, pred2))
    output.write("AP#%f: %f\n"%(num_ratio, AP2))
    output.write("LAP#%f: ")
    for num in LAP2:
        output.write("%f "%(num))
    output.write("\n")

output.close()