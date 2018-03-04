
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import ClaimsData
import turtle


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

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
        n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda:ClaimsData.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:ClaimsData.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    
    predict_x = {
        'NumberofViolations': [3,0,2,6],
        'NumberofAccidents': [0,4,5,6],
        'NumberofOpenComplaints': [2,0,5,6]
    }

    predictions = classifier.predict(
        input_fn=lambda:ClaimsData.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    turtle.screensize()
    turtle.setup(width = 1.0, height = 1.0)
    turtle.home()
    turtle.penup()
    turtle.left(60)
    turtle.forward(100)

    turtle.pensize(30)
    turtle.fillcolor("RED")
   
    x=-600
    y=250
    
    
    

    for pred_dict in predictions:
        template = ('\nPrediction is "{}" ({:.1f}%)')
        print(pred_dict)
       
        
        class_id = pred_dict['class_ids'][0]
        print (class_id)
        probability = pred_dict['probabilities'][class_id]
        
        y=y-30;
        turtle.goto(x,y)
        
        turtle.write(template.format(ClaimsData.Driver[class_id],
                              100 * probability),font=("Arial", 20, "normal"))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
