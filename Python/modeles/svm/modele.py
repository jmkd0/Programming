import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mode
from itertools import combinations
import cvxopt #pip install cvxopt
import matplotlib.pyplot as plt

class SVMBase1:
    def __init__(self, learning_rate=0.001, _lambda=0.01, nb_epoch=1000):
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.nb_epoch = nb_epoch
        self.W = None
        self.b = None
    def fit_base(self, X, y):
        nb_rows, nb_feature = X.shape
        #Gradiant descent
        self.W = np.zeros(nb_feature)
        self.b = 0

        for _ in range(self.nb_epoch):
            for i in range(nb_rows):
                if y[i]*(np.dot(self.W, X[i]) - self.b) >= 1 :
                    self.W -= self.learning_rate*(2*self._lambda * self.W)
                else:
                    self.W -= self.learning_rate * (2 * self._lambda * self.W - np.dot(X[i], y[i]))
                    self.b -= self.learning_rate * y[i]
    
    def predict_base(self, X):
        output = np.dot(X, self.W) - self.b
        return np.sign(output)
    
    def decision_base(self, X):
        result = np.dot(X, self.W) - self.b
        return result

class SVMBase2:
    def __init__(self, kernel="linear", C=1.0):
        self.kernel = kernel
        self.C = C
        self.X = None
        self.y = None
        self.alpha = None
        self.W = None
        self.b = None
        self.sv = None
        if kernel == "linear":
            self.kernel = self.linear
        elif kernel == 'gaussian':
            self.kernel = self.gaussian
        elif kernel == 'polynomial':
            self.kernel = self.polynomial
        
    def fit_base(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        n, m = self.X.shape
        K = np.zeros((n,n))
        for i in range(n):
            K[i,:] = self.kernel(self.X[i, np.newaxis], self.X)
        P = cvxopt.matrix(np.outer(self.y, self.y) * K)
        q = cvxopt.matrix(-np.ones((n,1)))
        G = cvxopt.matrix(np.vstack((np.eye(n)* - 1, np.eye(n))))
        h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        A = cvxopt.matrix(self.y, (1, n), 'd')
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options['show_progress'] = False
    
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(solution['x'])
        
        threshold = 1e-4
        self.sv = ((self.alpha > threshold) * (self.alpha < self.C)).flatten()
        self.W = np.dot(self.X[self.sv].T, self.alpha[self.sv] * self.y[self.sv, np.newaxis])
        self.b = np.mean(self.y[self.sv, np.newaxis] - self.alpha[self.sv] * self.y[self.sv, np.newaxis] * K[self.sv,self.sv][:, np.newaxis])
        
    def predict_base(self, X):
        n = X.shape[0]
        y_predict = np.zeros(n)
        for i in range(n):
            y_predict[i] = np.sum(self.alpha[self.sv] * self.y[self.sv, np.newaxis] * self.kernel(X[i], self.X[self.sv])[:,np.newaxis])
        
        return np.sign(y_predict + self.b)
    
    def decision_base(self, X):
        n = X.shape[0]
        y_predict = np.zeros(n)
        for i in range(n):
            y_predict[i] = np.sum(self.alpha[self.sv] * self.y[self.sv, np.newaxis] * self.kernel(X[i], self.X[self.sv])[:,np.newaxis])
        return y_predict + self.b
         
    def linear(self, X, Z):
        return np.dot(X, Z.T)
    
    def gaussian(self, X, Z, sigma=0.1):
        return np.exp(-np.linalg.norm(X-Z, axis=1)**2 / (2*(sigma**2)))
    
    def polynomial(self, X, Z, p=5):
        return (1 + np.dot(X, Z.T)) ** p
    
class SVM(SVMBase1, SVMBase2):
    def __init__(self, learning_rate=0.001, _lambda=0.01, nb_epoch=1000, decision='ova', kernel= None, C=1.0):
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.nb_epoch = nb_epoch
        self.labels = None
        self.nb_label = None
        self.kernel = kernel
        self.decision = decision
        self.classifiers = []
        self.class_pairs = []
        SVMBase1.__init__(self, learning_rate=learning_rate, _lambda=_lambda, nb_epoch=nb_epoch)
        SVMBase2.__init__(self, kernel=kernel, C=C)
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.labels = list(np.unique(y))
        self.nb_label = len(self.labels)
        y = np.array([self.labels.index(yi) for yi in y])
        if self.kernel == None:
            self.fit_classification(SVMClass = SVMBase1, X=X, y=y)
        else:
            self.fit_classification(SVMClass = SVMBase2, X=X, y=y)
                
    def fit_classification(self, SVMClass, X, y):
        y_unique = np.unique(y)
        if self.nb_label == 2:
            y = np.where(y == 1, 1, -1)
            SVMClass.fit_base(self, X=X, y=y)
        elif self.decision == 'ova':
            y_list = [np.where(y == label, 1, -1) for label in y_unique]
            for y_i in y_list:
                svm1 = SVMClass()
                svm1.fit_base(X, y_i)
                self.classifiers.append(svm1)
        elif self.decision == 'ovo':
            self.class_pairs = list(combinations(y_unique, 2))
            for class_pair in self.class_pairs:
                indexs = np.where((y == class_pair[0]) | (y == class_pair[1]))
                y_i = np.where(y[indexs] == class_pair[0], 1, -1)
                clf = SVMClass()
                clf.fit_base(X[indexs], y_i)
                self.classifiers.append(clf)
                
    def predict(self, X):
        result = self.predict_points(X)
        return np.array([self.labels[i] for i in result])
    
    def predict_points(self, X):
        X = np.array(X)
        if self.kernel == None:
            return self.predict_point_classification(predict = SVMBase1.predict_base, X=X)
        else:
            return self.predict_point_classification(predict = SVMBase2.predict_base, X=X)
    
    def predict_point_classification(self, predict, X):
        if self.nb_label == 2:
            result = predict(self, X=X)
            result = np.where(result == 1, 1, 0)
        elif self.decision == 'ova':
            result = self.predict_multiclass_ova(X)
        elif self.decision == 'ovo':
            result = self.predict_multiclass_ovo(X)
        return result
        
    def predict_multiclass_ova(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))
        for idx, clf in enumerate(self.classifiers):
            predictions[:, idx] = clf.decision_base(X)
        # return the argmax of the decision function as suggested by Vapnik.
        return np.argmax(predictions, axis=1)

    def predict_multiclass_ovo(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))
        for idx, clf in enumerate(self.classifiers):
            class_pair = self.class_pairs[idx]
            prediction = clf.predict_base(X)
            predictions[:, idx] = np.where(prediction == 1, class_pair[0], class_pair[1])
        return mode(predictions, axis=1)[0].ravel().astype(int)
    
    def score(self, y_true, y_predict):
        return np.sum(y_true == y_predict) / len(y_true)
 
    def plot_contours(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if len(X[0,:]) > 2:
            print("Can't display graph...")
            return
        y = np.array([self.labels.index(yi) for yi in y])
        # plot the resulting classifier
        h = 0.01
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        points = np.c_[xx.ravel(), yy.ravel()]
        if self.nb_label == 2:
            Z = self.predict_points(points)
        elif self.decision == 'ova':
            Z = self.predict_multiclass_ova(points)
        elif self.decision == 'ovo':
            Z = self.predict_multiclass_ovo(points)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # plt the points
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        plt.show()
        