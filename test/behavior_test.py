import csv
import yaml
import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd
import os
from src.data.make_dataset import make_dataset
from src.data.preprocess import read_label_map

"""
This script will run behavior or sensitivity tests indicated in the params.yaml file.
As for now only the accuracy is saved into the tests.csv file for comparison.
"""

params = yaml.safe_load(open('params.yaml'))['test']
dataset_processed = params['dataset_processed']
datasets = params['datasets']
test_ind = params['tests']

test_sets = [params in datasets]
model = tf.keras.models.load_model('models/trained_model')


def get_classification_report(dataset, name):
    """This function creates confusion_matrix.json and  conf.png.

    Parameters:
    dataset (object): tf.Dataset object
    name (string): Name of test dataset

    Returns:
    results (string): Text summary of the precision, recall, F1 score for each class.
    """
    val_samples = os.path.join(dataset_processed, name, 'samples_test.csv')
    samples_df = pd.read_csv(val_samples)
    labels = samples_df['label']
    scores = model.predict(dataset)
    predictions = scores.argmax(axis=1)
    label_map = [i[0] for i in read_label_map("data/processed/label_map.pbtxt").items() if
                 i[1] in predictions or i[1] in labels.values]
    results = classification_report(labels.values, predictions, target_names=label_map, output_dict=True)
    return results


def test_blur(dataset_name):
    dataset, _, _ = make_dataset('test', [dataset_name], blur=True)
    results = get_classification_report(dataset, name=dataset_name)
    return results


def test_gamma(dataset_name):
    dataset, _, _ = make_dataset('test', [dataset_name], gamma=True)
    results = get_classification_report(dataset, name=dataset_name)
    return results


def test_hue(dataset_name):
    dataset, _, _ = make_dataset('test', [dataset_name], hue=True)
    results = get_classification_report(dataset, name=dataset_name)
    return results


def test_downsample(dataset_name):
    dataset, _, _ = make_dataset('test', [dataset_name], downsample=True)
    results = get_classification_report(dataset, name=dataset_name)
    return results


def test_red_tone(dataset_name):
    dataset, _, _ = make_dataset('test', [dataset_name], tone='red')
    results = get_classification_report(dataset, name=dataset_name)
    return results


def test_blue_tone(dataset_name):
    dataset, _, _ = make_dataset('test', [dataset_name], tone='blue')
    results = get_classification_report(dataset, name=dataset_name)
    return results


tests = {'blur': test_blue_tone,
         'gamma': test_gamma,
         'hue': test_hue,
         'downsample': test_downsample,
         'red_tone': test_red_tone,
         'blue_tone': test_blue_tone}

original = ['original']
test_names = [test for test in test_ind]
accuracies = [original + test_names]

for i, test_set in enumerate(datasets):
    accuracy = []
    ds, _, _ = make_dataset('test', [test_set])
    results = get_classification_report(ds, name=test_set)
    accuracy.append(results['accuracy'])
    for test in tests.items():
        if test[0] in test_ind:
            results = test[1](test_set)
            accuracy.append(results['accuracy'])
    accuracies.append(accuracy)

wtr = csv.writer(open('report/tests.csv', 'w'), delimiter=',', lineterminator='\n')
for x in accuracies:
    wtr.writerow(x)
