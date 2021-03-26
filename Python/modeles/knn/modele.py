import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class KNN:
    def __init__(self, K = 5):
        self.K = K
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X = X.astype(np.int64)
        self.y = y
        self.labels = list(np.unique(self.y))
    
    def predict(self, X):
        X = np.array(X)
        #X = X.astype(np.int64)
        results = []
        for x in X:
            results.append(self.predict_element(x))
        return np.array(results)
    
    def predict_element(self, a):
        #Calculate distance from x to all element in the dataset
        list_dist = []
        for x, y in zip(self.X, self.y):
            dist = ((a - x) ** 2).sum()
            list_dist.append([dist, y])
        list_dist = sorted(list_dist, key=lambda li: li[0]) #sort the list according to the distance at the first column
        k_nearest = list_dist[:self.K] #Get the K first smallest datas
        y = np.array(k_nearest)[:,1]
        labels, counts = np.unique(y, return_counts=True) 
        best_label = labels[np.argmax(counts)]
        return best_label
    
    def score(self, y_true, y_predict):
        return np.sum(y_true == y_predict) / len(y_true)
    
    def plot_contours(self, X, y):
        X = np.array(X)
        if len(X[0,:]) > 2:
            print("Can't display graph...")
            return
        y = np.array(y)
        # plot the resulting classifier
        h = 0.01
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        points = np.c_[xx.ravel(), yy.ravel()]

        Z = self.predict(points)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # plt the points
        #plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        plt.show()

