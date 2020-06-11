# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:14:26 2020

@author: Apostolos
"""
from sklearn.cluster import KMeans
import numpy as np
class RBF:
    # k :number of rbf neuro
    # bias : boolean , if it's true then apply bias to hidden layer neurons
    # kmeans : boolean , if it's true , caculates the center with kmeans , else chose them randomly
    def __init__(self, k=5,bias=True,batch=1,kmeans=True):
        self.k = k
        self.kmeans = kmeans
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
        self.bias = bias
        self.batch=batch
    # Gaussian function for rbf neurons
    def rbf(self,xi, c, s):
        return np.exp(-1 / (2 * s**2) * np.linalg.norm(xi-c)**2)
    # function that uses kmeans to find centers of rbf neurons
    def CentersWithKmeans(self,k,X):
        model=KMeans(n_clusters=k)
        model.fit(X)
        return model.cluster_centers_
    #function that chose randomly centers for rbf neurons
    def RandomCenters(self,n):
        centers=[]
        for i in range(self.k):
            centers.append(np.random.randn(n))
        return centers
    # fit function , implements the training proccess
     # lr :learning rate
    def fit(self, X, y,epochs=20,lr=0.01,showAcc=True):
        
        if self.kmeans :
            self.centers= self.CentersWithKmeans(self.k,X)
        else:
            self.centers=self.RandomCenters(len(X[0]))
            
        d =[]
        for c1 in self.centers:
            for c2 in self.centers:
                d.append(np.linalg.norm(c1 - c2))
        Max = max(d)
        self.stds = Max / np.sqrt(2*self.k)
        print(self.stds)
        # training
        for epoch in range(epochs):
            sum=0
            sumerror=0
            for i in range(X.shape[0]):
                # a keeps the output from hidden layer
                at = np.array([self.rbf(X[i], c, self.stds) for c in self.centers])
                
                # F keep the output of Output layer
                if self.bias :
                    F = at.T.dot(self.w) + self.b
                else:
                    F = at.T.dot(self.w) 
                # squeare loss
                loss = (y[i] - F).flatten() ** 2
                sum+=loss
     
                # backward pass
                error = (y[i] - F)
                sumerror+=error
                # update
                if i% self.batch==0 or i==X.shape[0]-1:
                    self.w = self.w + lr * at * sumerror/self.batch
                    if self.bias : 
                        self.b = self.b + lr * sumerror/self.batch
                    sumerror=0
            if showAcc:    
                print("epoch : "+str(epoch)+" ",end=" ")
                print(" train acc: "+str(self.eval(X,y))+" loss: "+str(sum/X.shape[0]))
            else : 
                print("epoch : "+str(epoch)+" ")
    def predictOne(self,x):
        at = np.array([self.rbf(x, c, self.stds) for c in self.centers])
        if self.bias :
            F = at.T.dot(self.w) + self.b
        else :
            F = at.T.dot(self.w)
        if abs(F) < abs(F-1):
            return 0 
        return 1
        
        
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            at = np.array([self.rbf(X[i], c, self.stds) for c in self.centers])
            if self.bias :
                F = at.T.dot(self.w) + self.b
            else :
                F = at.T.dot(self.w)
            y_pred.append(F)
        return np.array(y_pred)
    # X data
    # Y labels [0-1]
    # returns the presentage of how accurate are the predictions of the Rbf neural network
    def eval(self,X,Y):
        preds=self.predict(X)
        c=0
        for i in range(X.shape[0]):
            if abs(preds[i]- Y[i]) < abs(preds[i]-(1-Y[i])):
                c+=1
        return c/X.shape[0]
        
        
        
        
        
        
        
        
    
        