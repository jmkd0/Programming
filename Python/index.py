import pandas as pd
import numpy as np
import json

productdict = {
    "Design": ["Iphone", "Samsung", "Dell", "HP"],
    "Price": [30, 20, 15, 34],
    "Codeclt": ["A", "A", "C", "D"],
    
}
clientdict = {
    "Code_clt": ["A", "B", "C", "D"], 
    "Name": ["John", "Pierre", "Dupont", "Malik"],
    "Adress": ["15 rue marie", "7 rue sentier", "32 rue Michelet", "15 Boulevard"]
    }
clientjson = """[
    {
        "Codeclt":"A",
        "Name":"John",
        "Adress":"15 rue marie"
    },
    {
        "Codeclt": "B",
        "Name": "Pierre",
        "Adress": "7 rue sentier"
    },
    {
        "Codeclt": "C",
        "Name": "Dupont",
        "Adress": "32 rue Michelet"
    },
    {
        "Codeclt": "E",
        "Name": "Johnson",
        "Adress": "15 Boulevard"
    }
]"""
#Read python Dictionnary to Pandas
clientdict = pd.DataFrame.from_dict(clientdict)
productdict = pd.DataFrame.from_dict(productdict)
clientjson1 = pd.read_json(clientjson)
#print(clientdict)
#print(clientjson1)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
productdict = pd.DataFrame.from_dict(productdict)
#productdict = dt
print(productdict)
for column in ["Codeclt","Design"]:
    productdict[column] = label_encoder.fit_transform(productdict[column])

print(productdict)
for column in ["Codeclt","Design"]:
    productdict["Design"] = label_encoder.inverse_transform(productdict[column])
#productdict["Design"] = label_encoder.inverse_transform(productdict["Design"])
print(productdict)
#Get row in pands where
"""
from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
print(productdict)
productdict = pd.DataFrame.from_dict(productdict)


for col in productdict.columns:
    productdict.loc[:,col] = Encoder.fit_transform(productdict.loc[:,col])
print(productdict)
for col in productdict.columns:
    productdict.loc[:,col] = Encoder.inverse_transform(productdict.loc[:,col])

print(productdict)

#for column in ["Design", "Codeclt"]:
#    productdict.loc[:,column] = Encoder.fit_transform(productdict.loc[:,column])
#
#print(productdict)
#for column in ["Design", "Codeclt"]:
#    productdict.loc[:,column] = Encoder.inverse_transform(productdict.loc[:,column])

#productdict['Design'] = Encoder.fit_transform(productdict['Design'])
#productdict['Codeclt'] = Encoder.fit_transform(productdict['Codeclt'])
#print(productdict)
#result
#productdict['Design'] = Encoder.inverse_transform(productdict['Design'])
#productdict['Codeclt'] = Encoder.inverse_transform(productdict['Codeclt'])
"""
