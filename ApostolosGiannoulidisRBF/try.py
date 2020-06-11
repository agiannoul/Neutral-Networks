# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:52:53 2020

@author: Apostolos
"""

from keras.datasets import mnist
import rbf
import time

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
print("Time : ",time.process_time() - start)

print("Train results : ",end=" ")
print(r.eval(x_train,y_train))
print("Testing results : ",end=" ")
print(r.eval(x_test,y_test))









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






