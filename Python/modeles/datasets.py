from sklearn import datasets
import numpy as np
import pandas as pd
def load_data(k):
    if k == 0:
        X = np.array([[4, 2], [4, 3], [5, 1], [5, 2],
                     [5, 3], [6, 1], [6, 2], [9, 4],
                     [9, 7], [10, 5], [10, 6], [11, 6]])
        y = np.array([1,1,1,1,1,1,1,-1,-1,-1,-1,-1])
        #['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no','no', 'no']
    elif k == 1:
        X = np.array([[1, 6], [1, 7], [2, 5], [2, 8],
                     [4, 2], [4, 3], [5, 1], [5, 2],
                     [5, 3], [6, 1], [6, 2], [9, 4],
                     [9, 7], [10, 5], [10, 6], [11, 6],
                     [5, 9], [5, 10], [5, 11], [6, 9],
                     [6, 10], [7, 10], [8, 11]])
        y = np.array([1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3,
                     4, 4, 4, 4, 4, 4, 4])
    elif k == 2:
        X, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    elif k == 3:
        iris = datasets.load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data["label"] = iris.target
        X, y  = data.drop("label", axis=1), data["label"].values
    elif k == 4:
        N = 50
        K = 2
        D = 2
        X = np.zeros((N * K, D))  # data matrix (each row = single example)
        y = np.zeros(N * K)  # class labels

        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j

        # lets visualize the data:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

        y[y == 0] -= 1
    elif k == 5:
        dat = datasets.load_breast_cancer()
        data = pd.DataFrame(dat.data, columns=dat.feature_names)
        data["label"] = dat.target
        X = data.drop('label', axis=1).values# Input features (attributes)
        y = data['label'].values
    return X, y
        
