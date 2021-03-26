import sys
sys.path.append('../..')
from modeles.datasets import load_data
from modele import SVM
import numpy as np
import pandas as pd



X, y = load_data(0)


#Kernel
modele = SVM(kernel='linear', decision='ova')
modele.fit(X, y)
predictions = modele.predict(X)
score = modele.score(y, predictions)
#modele.plot_contours(X, y)
print("y in:", y)
print("y out ",predictions)
print("Accuracy :",score)