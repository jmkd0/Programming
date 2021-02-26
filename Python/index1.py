import pandas as pd
import json 
import numpy as np

productdict = {
    "Design": ["Fot", "Faible", "Dell", "HP"],
    "Potentiel": [30, 20, 15, 34],
    "Codeclt": ["A", "A", "C", "D"]
}
productdict = pd.DataFrame.from_dict(productdict)
print(productdict)



from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pandas as pd
productdict = {
    "Design": ["Fot", "Faible", "Dell", "HP"],
    "Potentiel": [1, 2, 2, 6],
    "Codeclt": ["A", "A", "C", "D"]
}
productdict = pd.DataFrame.from_dict(productdict)
print(productdict)
print(productdict.dtypes)
for col in productdict.columns:
    if productdict.dtypes[col] == object :
        productdict.loc[:,col] = label_encoder.fit_transform(productdict.loc[:,col])
print(productdict)
print(productdict.dtypes)
for col in productdict.columns:
    if productdict.dtypes[col] == 'int32' :
        productdict.loc[:,col] = label_encoder.inverse_transform(productdict.loc[:,col])

print(productdict)
#request


productdict['Design'] = label_encoder.fit_transform(productdict['Design'])
print(productdict)
#result
productdict['Design'] = label_encoder.inverse_transform(productdict['Design'])




from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
#print(dt.dtypes)
for col in dt_.columns:
    if dt_.dtypes[col] == object:
        dt_.loc[:,col] = label_encoder.fit_transform(dt_.loc[:,col].astype(str))