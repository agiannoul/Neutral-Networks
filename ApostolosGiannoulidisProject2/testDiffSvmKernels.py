# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:39:31 2019

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
    




print("--------Start---------")


import time
###
for kernel in ('linear', 'poly', 'rbf'):
    print("\n")
    print(kernel)
    start = time.process_time()
    C=10.0 # We chose these values for c anf gamma after the execution searchGandC.py
    gamma = 0.01
    n=random.randrange(len(x_train)-5000)
    n1=random.randrange(len(x_test)-1000)      
    clf = SVC(kernel=kernel,C=C, gamma=gamma)
    clf.fit(x_train[n:n+5000], y_train[n:n+5000])
    predictions = clf.predict(x_test[n1:n1+1000])
    print("Test acc: %f" %(accuracy_score(y_test[n1:n1+1000], predictions)))
    predictions = clf.predict(x_train[n:n+5000])
    print("Train acc: %f" %(accuracy_score(y_train[n:n+5000], predictions)))
    print("Time : ",time.process_time() - start)





