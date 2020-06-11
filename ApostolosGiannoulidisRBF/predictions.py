# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:52:53 2020

@author: Apostolos
"""

from keras.datasets import mnist
import rbf
import time
import matplotlib.pyplot as plt
(x_train1, y_train), (x_test1, y_test) = mnist.load_data()



x_train = x_train1.reshape((-1, 28*28))/255
x_test = x_test1.reshape((-1, 28*28))/255

#x_train= x_train[0:10000]
#y_train= y_train[0:10000]
#x_test =x_test[0:3000]
#y_test =y_test[0:3000]

from sklearn.decomposition import PCA
print("PCA :",end=" ")
pca = PCA(0.8)

pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

print(pca.n_components_)
for i in range(len(y_train)):
    y_train[i]=y_train[i] % 2

for i in range(len(y_test)):
    y_test[i]=y_test[i] % 2

r = rbf.RBF(k=5,bias=True,kmeans=True)
start = time.process_time()
r.fit(x_train,y_train,showAcc=True,lr=0.01, epochs=13)



pos=-1
for i in range(0,10000):
    k= x_test[i]
    
    pr=r.predictOne(k)
    if pr==y_test[i]:
        pos=i
        break
if pos != -1 :
    k= x_test[pos]
    pr=r.predictOne(k)
    print(" Prediction : ",end=" ")
    print(pr)
    print(" Real value : ",end=" ")
    print(y_test[pos])


    plt.imshow(x_test1[pos], cmap=plt.get_cmap('gray'))
    plt.show()


pos=-1
for i in range(0,10000):
    k= x_test[i]
    
    pr=r.predictOne(k)
    if pr!=y_test[i]:
        pos=i
        break
if pos != -1 :
    k= x_test[pos]
    pr=r.predictOne(k)
    print(" Prediction : ",end=" ")
    print(pr)
    print(" Real value : ",end=" ")
    print(y_test[pos])


    plt.imshow(x_test1[pos], cmap=plt.get_cmap('gray'))
    plt.show()

