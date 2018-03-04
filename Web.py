from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import Webscreen
import ClaimsData
import turtle
import pandas as pd
import os

from flask import Flask, render_template,request

app = Flask(__name__)

@app.route('/')
def homescreen():
    return render_template('index.html')


# To serve the get request from the home page and sends the report to web UI.
@app.route('/prediction',methods=['GET', 'POST'])
def reportscreen():

    if request.method == 'POST':
        f = request.files['file']
        print(f)
        
        predictions,predict,name=Webscreen.predict()
        claimdata=predict.to_dict(orient="records")
        temp=0
        predictions_loc=predictions
        for pred_dict in predictions_loc:
            template = ('\nPrediction is "{}" ({:.1f}%)')
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            claimdata[temp]["probability"]=probability*100
            claimdata[temp]["Name"]=name[temp]
            claimdata[temp]["Prediction"]=ClaimsData.Driver[class_id]
            temp=temp+1

        print(claimdata)
        
        return render_template('report.html', data=claimdata)
        
    
    



if __name__ == '__main__':
    app.run()
    
