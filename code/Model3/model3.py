# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:16:30 2016

@author: lenovo
"""

import numpy
from sklearn.cluster import KMeans

def KL_divergence(p,q,dim,gamma): #KL散度计算公式
    res = 0
    for i in range(dim):
        res = res + (p[i]*numpy.log((p[i]+gamma)/(q[i]+gamma)) + q[i]*numpy.log((q[i]+gamma)/(p[i]+gamma)))/2
    return res

def cosSim(doc_a,doc_b): #余弦相似度计算公式
    return numpy.dot(doc_a,doc_b)/(numpy.sqrt(abs(numpy.dot(doc_a,doc_a)))*numpy.sqrt(abs(numpy.dot(doc_b,doc_b))))

def cal_weight(label_cluster):
    weight = []
    for label in label_cluster:
        sum_ = 0
        cnt = 0
        for line in label:
            tmp = numpy.std(line)
            sum_ = sum_ + tmp
            cnt = cnt+1
        weight_num = sum_/cnt
        weight.append(weight_num)
    return weight

def Tar_docClustering(X_target_k,k1,Y_target_k,X_target,threhold): #文档聚类
    clusters = []
    label_clusters = []
    kmeans = KMeans(n_clusters=k1, random_state=0).fit(X_target_k)
    for i in range(k1):
        doc_cluster = []
        tmp_labels = []
        cnt = 0
        for label in kmeans.labels_:
            if label == i:
                doc_cluster.append(X_target_k[cnt,:])
                tmp_labels.append(Y_target_k[cnt,:])
            cnt = cnt+1
        if len(doc_cluster) > 0 and len(tmp_labels) > 0:
            clusters.append(doc_cluster)
            label_clusters.append(tmp_labels)
    k1 = len(clusters)
    for doc_i in X_target: #对于target集合里的每一篇文档，与当前所有簇里的中心文档进行余弦相似度的比较，加入最相似的簇
        max_ = 0
        idx = -1
        for i in range(k1):
            tmp_cluster = clusters[i]
            sim = 0
            cnt = 0
            for doc_j in tmp_cluster:
                sim = sim + cosSim(doc_i,doc_j)
                cnt = cnt + 1
            if cnt != 0:
                sim = sim/cnt
            if sim > max_:
                max_ = sim
                idx = i
        if max_ > threhold:
            clusters[idx].append(doc_i)
            #clusters[idx].sort(key=lambda x:x[0] , reverse=True) #每个簇内部按照相似度大小维护顺序
    return clusters,label_clusters

def Src_docClustering(X_source,k2,Y_source): #文档聚类
    clusters = []
    label_clusters = []
    kmeans = KMeans(n_clusters=k2, random_state=0).fit(X_source)
    for i in range(k2):
        doc_cluster = []
        tmp_labels = []
        cnt = 0
        for label in kmeans.labels_:
            if label == i:
                doc_cluster.append(X_source[cnt,:])
                tmp_labels.append(Y_source[cnt,:])
            cnt = cnt+1
        if len(doc_cluster) > 0 and len(tmp_labels) > 0:
            clusters.append(doc_cluster)
            label_clusters.append(tmp_labels)
    return clusters,label_clusters
            
def discover(src_clusters,src_label_clusters,tar_clusters,label_clusters,dim,gamma): #标签聚类
    t_clusters = src_clusters
    t_label_clusters = src_label_clusters
    cntt = 0
    for label_i in tar_label_clusters:
        min_ = 100000000
        idx = -1
        cnt = 0
        label_i = numpy.mean(label_i,axis=0)        
        for label_j in src_label_clusters:
            label_j = numpy.mean(label_j,axis=0)
            div = KL_divergence(label_i,label_j,dim,gamma)
            if div < min_:
                min_ = div
                idx = cnt
            cnt = cnt+1
        for i in range(len(tar_label_clusters[cntt])):
            t_clusters[idx].append(tar_clusters[cntt][i])
            t_label_clusters[idx].append(tar_label_clusters[cntt][i])
        cntt = cntt+1
    return t_clusters,t_label_clusters


def labelChange(Y):
    Y_ = numpy.argmax(Y,1)
    return Y_

def CostFunction2(theta , t_clusters , label_clusters ,weight ,lam, dim): #基于文档簇的多元逻辑回归分类
    grad = numpy.zeros(theta.shape)
    cnt = 0
    J = 0
    label_clusters = numpy.array(label_clusters)
    num = label_clusters.shape[0]
    for cluster in t_clusters:	#求代价函数
        cluster = numpy.array(cluster)
        size1 = cluster.shape[0]
        size2 = len(label_clusters[cnt])
        mean_label = numpy.argmax(numpy.mean(numpy.array(label_clusters[cnt]),axis=0))
        Y = numpy.argmax(numpy.array(label_clusters[cnt]),1)
        for i in range(size1-size2):
            numpy.append(Y,mean_label)
        size = cluster.shape[0]
        temp = numpy.exp(numpy.dot(cluster,theta.transpose()))
        neg_like = numpy.zeros(size)
        J_ = 0
        for i in range(size):
            j = Y[i]
            neg_like[i] = numpy.log(temp[i][j] / temp[i].sum())
            J_ = J_ + numpy.sum(neg_like)/(-size)
        J = J + weight[cnt] * J_     
        cnt = cnt+1
    reg = numpy.sum(theta*theta)
    J = J + lam*reg/(2*num)
    cnt = 0
    for cluster in t_clusters:	#求梯度
        cluster = numpy.array(cluster)
        size = cluster.shape[0]
        temp = numpy.exp(numpy.dot(cluster,theta.transpose()))
        Y = numpy.argmax(numpy.mean(numpy.array(label_clusters[cnt]),axis=0))
        for j in range(dim):
            var = numpy.zeros(grad.shape[1])
            for i in range(size):
                if(j == Y):
                    var = var + cluster[i]*(1-temp[i][j]/temp[i].sum())
                else:
                    var = var + cluster[i]*(-temp[i][j]/temp[i].sum())
            grad[j] = grad[j] + (weight[cnt]*var)/(-size)
        cnt = cnt+1
    grad = grad + lam*theta/num
    return J,grad	#返回代价函数值与梯度值

def corr_cal(x,y):
    size = x.shape[0]
    corr = numpy.corrcoef(x,y)[:size,size:]
    corr = numpy.diag(numpy.nan_to_num(corr))
    return numpy.sum(corr)/numpy.count_nonzero(corr)

def Lcorr_cal(x,y):
    size = x.shape[1]
    corr = numpy.corrcoef(x,y,rowvar=0)[:size,size:]
    corr = numpy.diag(corr)
    return corr
    
def predict(theta,X,Y,dim):	#根据权重进行预测
    size = X.shape[0]
    temp = numpy.dot(X,theta.transpose())
    neg_like = numpy.zeros([size,dim])
    for i in range(size):
        for j in range(dim):
            neg_like[i][j] = numpy.exp(temp[i][j]) / numpy.sum(numpy.exp(temp[i]))
    AP = corr_cal(neg_like,Y)  
    LAP = Lcorr_cal(neg_like,Y)
    Y_p = numpy.array(labelChange(neg_like))
    Y = numpy.array(labelChange(Y))
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
test_num = 0
iter_times = 1000
gamma = 0.0001
threhold = 0.25

best_acc = 0
best_AP = -1
best_LAP = []

output = open('res__' , 'w' , encoding= 'utf8')

for num_ratio in [1/8]:
    num = round((a.shape[0]-train_num)*num_ratio)
    
    for k_ratio in [1]:
        k1 = round(k_ratio*num)
        k2 = round(k_ratio*X_source.shape[0])
        for lam in [0.5]:        
            X_training = X_source
            Y_training = Y_source
            X_testing_k = X_target[:num]
            Y_testing_k = Y_target[:num]
            X_testing = X_target[num:]
            Y_testing = Y_target[num:]
            word_num = X_training.shape[1]

            tar_clusters,tar_label_clusters = Tar_docClustering(X_testing_k,k1,Y_testing_k,X_target,threhold)
            src_clusters,src_label_clusters = Src_docClustering(X_training,k2,Y_training)
            
            t_clusters,t_label_clusters = discover(src_clusters,src_label_clusters,tar_clusters,tar_label_clusters,dim,gamma)
            weight = cal_weight(t_label_clusters)  
            
            theta = numpy.zeros([dim,word_num])
            
            cnt = 0
            last_J = -1
            while cnt < iter_times:
                J,grad = CostFunction2(theta , t_clusters , t_label_clusters ,weight, lam, dim)
                if(numpy.abs(J-last_J) < 0.000001):
                    break
                theta = theta - step*grad
                cnt = cnt + 1
                last_J = J
            
            AP1,LAP1,res1 = predict(theta,X_training,Y_training,dim)
            AP2,LAP2,res2 = predict(theta,X_testing,Y_testing,dim)
            if res2 > best_acc:
                best_acc = res2
            if AP2 > best_AP:
                best_AP = AP2
                best_LAP = LAP2

            print("Accuracy#%f: %f\n"%(num_ratio, res2))
            print("AP#%f: %f\n"%(num_ratio, AP2))                
                
    output.write("Accuracy#%f: %f\n"%(num_ratio, best_acc))
    output.write("AP#%f: %f\n"%(num_ratio, best_AP))
    output.write("LAP#%f: ")
    for num in best_LAP:
        output.write("%f "%(num))
    output.write("\n")

output.close()
