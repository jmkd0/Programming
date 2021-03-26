import numpy as np
from collections import Counter
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, * , value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_rows_split=2, max_depth=100, nb_features=None):
        self.min_rows_split = min_rows_split
        self.max_depth = max_depth
        self.nb_features = nb_features
        self.root = None
    
    def fit(self, X, y):
        X = np.array(X)
        self.nb_features = X.shape[1] if not self.nb_features else min(self.nb_features, X.shape[1])
        self.root = self.build_tree(X,y)
    
    def build_tree(self, X, y, depth=0):
        nb_rows, nb_feats = X.shape
        nb_labels = len(np.unique(y))
        if depth >= self.max_depth or nb_labels == 1 or nb_rows < self.min_rows_split:
            leaf_value = Counter(y).most_common(1)[0][0] #Get the most common label in all rows
            return Node(value=leaf_value)
        
        feats_index = np.random.choice(nb_feats, self.nb_features, replace=False) #create self.nb_features random list between 0-nb_feats with no repeat
        #greedy search
        best_feat, best_thresh = self.best_criteria(X, y, feats_index)
        #Generate split
        left_index = np.argwhere(X[:,best_feat] <= best_thresh).flatten()
        right_index = np.argwhere(X[:,best_feat] > best_thresh).flatten()
        #Build Child tree
        left = self.build_tree(X[left_index,:], y[left_index], depth+1)
        right = self.build_tree(X[right_index,:], y[right_index], depth+1)
        return Node(best_feat, best_thresh, left, right)

    
    def best_criteria(self, X, y, feats_index):
        best_gain = -1
        split_col, split_thresh = None, None
        #Loop over features index and then over each threshold
        for col_index in feats_index:
            X_column = X[:,col_index]
            thresholds = np.unique(X_column)
            for  thresh in thresholds:
                gain = self.information_gain(X_column, y, thresh) # Calculate information Gain
                if gain > best_gain:
                    best_gain = gain
                    split_col = col_index
                    split_thresh = thresh
        return split_col, split_thresh
    
    def information_gain(self, x, y, thresh):
        #parent entropy
        parent_entropy = self.entropy(y)
        #Generate split
        left_index = np.argwhere(x <= thresh).flatten()
        right_index = np.argwhere(x > thresh).flatten()

        len_l, len_r = len(left_index), len(right_index)
        if len_l == 0 or len_r == 0:
            return 0
        #Weighted average for childs
        len_y = len(y)
        left_entropy = self.entropy(y[left_index])
        right_entropy = self.entropy(y[right_index])
        childs_entropy = (len_l/len_y)*left_entropy + (len_r/len_y)*right_entropy
        #Info gain
        info_gain = parent_entropy - childs_entropy
        return info_gain
        

    def entropy(self, label):
        hist = np.bincount(label)
        proba = hist / len(label)
        entropy = -np.sum([p * np.log2(p) for p in proba if p > 0])
        return entropy

    #Prediction function
    def predict(self, X):
        X = np.array(X)
        return np.array([self.traverse_tree(x, self.root) for x in X])
    
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    #Score of prediction
    def score(self, y_true, y_predict):
        return np.sum(y_true == y_predict) / len(y_true)
    
    def drow_graph(self, DT, features, labels):
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(DT, feature_names=features,  
                           class_names=labels.astype(str),
                           filled=True)
        fig.show()

