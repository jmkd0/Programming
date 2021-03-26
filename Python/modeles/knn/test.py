import sys
sys.path.append('../..')
from modeles.datasets import load_data
from modele import KNN

X, y = load_data(0)


knn = KNN(K=5)
knn.fit(X, y)
y_predict = knn.predict(X)
score = knn.score(y, y_predict)
#knn.plot_contours(X, y)
print("y in: ", y)
print("y out: ", y_predict)
print("The accuracy is: ", score)