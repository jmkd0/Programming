import sys
sys.path.append('../..')
from modeles.datasets import load_data
from modele import DecisionTree
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import pandas as pd


X, y = load_data(5)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=1234)

DT = DecisionTree(max_depth=3)
#DT = DecisionTreeClassifier(max_depth=2)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
score = DT.score(y_test, y_pred)
print("The accuracy is: ", score)
#DT.drow_graph(DT, data.columns, data['label'].values)