from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import ClaimsData
import turtle
import pandas as pd
import os


from flask import Flask, render_template

currentWorkingDirectory = os.getcwd()

CSV_COLUMN_NAMES = ['NumberofViolations', 'NumberofAccidents','NumberofOpenComplaints','UpdatedCoverage',
                    'Driver']

def predict():
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = ClaimsData.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
       
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 2 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda:ClaimsData.train_input_fn(train_x, train_y,100),steps=1000)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:ClaimsData.eval_input_fn(test_x, test_y,100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    predict = pd.read_csv(currentWorkingDirectory+"\ClaimsData_predict.csv", names=CSV_COLUMN_NAMES, header=0)

    predict_x, name = predict, predict.pop("Driver")

    predictions = classifier.predict(
        input_fn=lambda:ClaimsData.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=100))

    return predictions,predict,name
 
    
