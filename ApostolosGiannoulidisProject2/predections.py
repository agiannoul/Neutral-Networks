# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:15:45 2019

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
    


# train svm 
C=10.0
gamma = 0.01
            
print(" C = %f , G = %f " %(C,gamma)) 
clf = SVC(C=C, gamma=gamma)
clf.fit(x_train, y_train)



import matplotlib.pyplot as plt


pos=-1
for i in range(0,10000):
    k= x_test[i]
    
    pr=clf.predict([k])
    if pr[0]==y_test[i]:
        pos=i
        break
if pos != -1 :
    k= x_test[pos]
    pr=clf.predict([k])
    print(" Prediction : ",end=" ")
    print(pr[0])
    print(" Real value : ",end=" ")
    print(y_test[pos])


    plt.imshow(x_test1[pos], cmap=plt.get_cmap('gray'))
    plt.show()


pos=-1
for i in range(0,10000):
    k= x_test[i]
    
    pr=clf.predict([k])
    if pr[0]!=y_test[i]:
        pos=i
        break
if pos != -1 :
    k= x_test[pos]
    pr=clf.predict([k])
    print(" Prediction : ",end=" ")
    print(pr[0])
    print(" Real value : ",end=" ")
    print(y_test[pos])


    plt.imshow(x_test1[pos], cmap=plt.get_cmap('gray'))
    plt.show()


