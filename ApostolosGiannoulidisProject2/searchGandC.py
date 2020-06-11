# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:24:15 2019

@author: Apostolos
"""

from keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random 


(x_train1, y_train), (x_test1, y_test) = mnist.load_data()



x_train = x_train1.reshape((-1, 28*28))/255
x_test = x_test1.reshape((-1, 28*28))/255




from sklearn.decomposition import PCA

pca = PCA(0.9)

pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(pca.n_components_)
for i in range(len(y_train)):
    y_train[i]=y_train[i] % 2

for i in range(len(y_test)):
    y_test[i]=y_test[i] % 2



C_d=[0.01,0.1,1,10]
G_d=[0.001,0.01,0.1,1,10]

for c in C_d:
    for g in G_d:
        n=random.randrange(len(x_train)-1200)
        n1=random.randrange(len(x_test)-400)
        clf = SVC(C=c , gamma=g)
        clf.fit(x_train[n:n+1200],y_train[n:n+1200])
        predicts=clf.predict(x_test[n1:n1+400])
        print(accuracy_score(y_test[n1:n1+400],predicts),end=" ")
        print(" C = %f , G = %f " %(c,g))

