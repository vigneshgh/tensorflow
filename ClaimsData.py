import pandas as pd
import tensorflow as tf
import os

currentWorkingDirectory = os.getcwd()


CSV_COLUMN_NAMES = ['NumberofViolations', 'NumberofAccidents','NumberofOpenComplaints','UpdatedCoverage',
                    'Driver']
Driver = ['Valid Claim', 'Fraudulent Claim']

def path_dataset():
    train_path = currentWorkingDirectory+"\ClaimsData_training.csv"
    test_path = currentWorkingDirectory+"\ClaimsData_test.csv"

    return train_path, test_path

def load_data(columntopop_name='Driver'):
    """Returns the claims dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = path_dataset()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
   
    train_x, train_y = train, train.pop(columntopop_name)   

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(columntopop_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
