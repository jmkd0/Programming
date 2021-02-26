from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

@app.route("/",methods = ['POST', 'GET'])
def home():
    a =2
    return render_template("index.html", a=a)

@app.route('/send_datas',methods = ['POST', 'GET'])
def processData():
    if request.method == "POST":
        get_value = request.form['invalue']
        identity = {
            "Name": get_value,
            "Age": 23 
        }
        productdict = {
            "Design": ["Iphone", "Samsung", "Dell", "HP"],
            "Price": [30, 20, 15, 34],
            "Codeclt": ["A", "A", "C", "D"]
        }
        productdict = pd.DataFrame.from_dict(productdict)
        print(productdict)
        #Send JSON
        json_objet = productdict.to_json(orient='records')
        json_row = productdict.to_json(orient='split')
        json_column = productdict.to_json(orient='columns')
        #Send List
        liste = productdict["Design"].values.tolist()
        print(liste)
        return render_template("result.html", response = identity, json_objet=json_objet, json_rows=json_row, json_columns=json_column, liste=liste) 
    return render_template("index.html") 

if __name__ == '__main__':
   app.run(debug = True, port='4004')

