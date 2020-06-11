# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:43:40 2019

@author: Apostolos
"""
from keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



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


start = time.process_time()
C=10.0
gamma = 0.01
            
print(" C = %f , G = %f " %(C,gamma)) 
clf = SVC(C=C, gamma=gamma)
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
print("Test acc: %f" %(accuracy_score(y_test, predictions)))
predictions = clf.predict(x_train)
print("Train acc: %f" %(accuracy_score(y_train, predictions)))
print("Time : ",time.process_time() - start)









from sklearn.neighbors import KNeighborsClassifier


start = time.process_time()

model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
model.fit(x_train, y_train)
print("Time for fit : ",time.process_time() - start)
print("Train acc k=3 =", model.score(x_train, y_train))

print("Test acc k=3  =", model.score(x_test, y_test))
print("Time : ",time.process_time() - start)


start = time.process_time()
model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
model.fit(x_train, y_train)

print("Time for fit : ",time.process_time() - start)

print("Train acc k=1 =", model.score(x_train, y_train))

print("Test acc k=1 =", model.score(x_test, y_test))
print("Time : ",time.process_time() - start)



from sklearn.neighbors import NearestCentroid

start = time.process_time()
model = NearestCentroid(metric='euclidean')
model.fit(x_train, y_train)
print("Time for fit : ",time.process_time() - start)

print("Train acc =", model.score(x_train, y_train))
print("Test acc =", model.score(x_test, y_test))

print("Time : ",time.process_time() - start)



