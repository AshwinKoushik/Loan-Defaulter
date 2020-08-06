# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:02:50 2020

@author: Ashwin
"""

import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    int_features=[float(x) for x in request.form.values()]
    final_features=np.array(int_features)
    final_features=final_features.reshape(1,-1)
    prediction=model.predict(final_features)
    
    output=round(prediction[0],13)
    return render_template('index.html', prediction_text='Loan predcition {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
     