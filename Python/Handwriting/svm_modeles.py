import numpy as np
import matplotlib.pyplot as plt
class SVM_1:
    def __init__(self, learning_rate=0.001, _lambda=0.01, nb_epoch=1000):
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.nb_epoch = nb_epoch
        self.W = None
        self.b = None
    def fit(self, X, y):
        _y = np.where(y <= 0, -1, 1)
        nb_rows, nb_feature = X.shape
        #Gradiant descent
        self.W = np.zeros(nb_feature)
        self.b = 0

        for _ in range(self.nb_epoch):
            for i in range(nb_rows):
                if _y[i]*(np.dot(self.W, X[i]) - self.b) >= 1 :
                    self.W -= self.learning_rate*(2*self._lambda * self.W)
                else:
                    self.W -= self.learning_rate * (2 * self._lambda * self.W - np.dot(X[i], _y[i]))
                    self.b -= self.learning_rate * _y[i]
    
    def predict(self, X):
        output = np.dot(X, self.W) - self.b
        return np.sign(output)

    def score(self, y_true, y_predict):
        return np.sum(y_true == y_predict) / len(y_true)

    def visualize_svm(self, W, b):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X[:,0], X[:,1], marker='o',c=y)
        #Get points
        x1 = np.amin(X[:,0])
        x2 = np.amax(X[:,0])

        #Determine ax+b=-1
        ya1 = (-W[0] * x1 + b - 1) / W[1] 
        ya2 = (-W[0] * x2 + b - 1) / W[1]

        yb1 = (-W[0] * x1 + b) / W[1]
        yb2 = (-W[0] * x2 + b) / W[1]

        yc1 = (-W[0] * x1 + b + 1) / W[1]
        yc2 = (-W[0] * x2 + b + 1) / W[1]

        #Plot lines
        ax.plot([x1, x2],[ya1, ya2], 'k--')
        ax.plot([x1, x2],[yb1, yb2], 'r')
        ax.plot([x1, x2],[yc1, yc2], 'k--')
        #Set axe minimum and maximum
        y_min = np.amin(X[:,1])
        y_max = np.amax(X[:,1])
        ax.set_ylim([y_min-3,y_max+3])

        plt.show()
    

from sklearn import datasets
X, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)

y = np.where(y == 0, -1, 1)
svm = SVM_1()
svm.fit(X, y)
predictions = svm.predict(X)
svm.visualize_svm(svm.W, svm.b)