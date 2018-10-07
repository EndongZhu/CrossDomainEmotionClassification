# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:53:53 2017

@author: lenovo
"""
import numpy 

def CostFunction(theta , X , Y , lam, dim):
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

def predict(theta,X,Y,dim):	#根据权重进行预测
    size = X.shape[0]
    temp = numpy.dot(X,theta.transpose())
    neg_like = numpy.zeros([size,dim])
    for i in range(size):
        for j in range(dim):
            neg_like[i][j] = numpy.exp(temp[i][j]) / numpy.sum(numpy.exp(temp[i]))
    Y_p = numpy.array(numpy.argmax(neg_like,1))
    cnt = 0    
    for i in range(size):
        if(Y_p[i] == Y[i]):
            cnt = cnt+1
    return Y_p,cnt/size

a = numpy.loadtxt("../whole",delimiter=",")
a = numpy.transpose(numpy.transpose(a[:,1:]))
b = numpy.loadtxt("../label_train",delimiter=",")
b = numpy.transpose(numpy.transpose(b[:,1:]))
c = numpy.loadtxt("../label_test",delimiter=",")
c = numpy.transpose(numpy.transpose(c[:,1:]))
d = numpy.concatenate((b,c))

a = (a>0) + 0

num = 256
X_source = a[:num]
Y_source = numpy.array(numpy.argmax(d,1))[:num]  
X_target = a[246:] 
Y_target = numpy.array(numpy.argmax(d,1))[246:]



tmp1 = sum(X_source)/num
tmp2 = sum(X_target)/1000

step = 5
dim = 2
tmp = (tmp2 > 0.001)&(tmp1>0.001)

ans = a[:,tmp]
a=a.compress(numpy.logical_not(tmp),axis=1)
X_source = a[0:num]
X_target = a[num:] 
feature_num = X_target.shape[1]
weights = []



cnt = 0
for scl_label in ans.T:							#训练主元，输出权重
    theta = numpy.zeros([2,feature_num])
    for i in range(100):
        J,grad = CostFunction(theta , X_target , scl_label[num:] , 0.1, dim)
        theta = theta - step*grad
    label,pred = predict(theta,X_target,scl_label[246:],dim)
    theta = theta[0,:]
    if cnt == 0:
        weights = [theta]
    else: 
        weights.append(theta)
    cnt = cnt + 1
    
weights = numpy.array(weights).T
U,sigma,VT=numpy.linalg.svd(weights)
scl_theta = U[:1000,:].T

a = numpy.loadtxt("whole",delimiter=",")
a = numpy.transpose(numpy.transpose(a[:,1:]))

X_source = numpy.concatenate((a[:num,:],numpy.dot(X_source,scl_theta)),axis=1)
X_target = numpy.concatenate((a[num:,:],numpy.dot(X_target,scl_theta)),axis=1)

feature_num = X_target.shape[1]
dim = 6
theta = numpy.zeros([dim,feature_num])
for i in range(1000):					#lr训练
    J,grad = CostFunction(theta , X_source , Y_source , 0.5, dim)
    print(J)
    theta = theta - step*grad
labell,pred1 = predict(theta,X_source,Y_source,dim)
labell,pred2 = predict(theta,X_target,Y_target,dim)
print(pred1)
print(pred2)