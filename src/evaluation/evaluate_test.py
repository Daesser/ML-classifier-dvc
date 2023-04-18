import yaml
import tensorflow as tf
import pandas as pd
import os
from src.data.make_dataset import make_dataset
from src.data.preprocess import read_label_map, preprocess
from evaluate import create_confusion_matrix
from src.data.sample import create_samples

"""
This script make predictions for test sets indicated in the params.yaml file (after preprocessing and sampling) and
save them into a csv file. It also creates confusion matrices. Sampling has to be done once as the samples are saved
into a csv file.
"""

data = yaml.safe_load(open('params.yaml'))['data']
params = yaml.safe_load(open('params.yaml'))['test']
path_processed = params['dataset_processed']
path_raw = params['dataset_raw']
datasets = params['datasets']

# Preprocessing and Sampling has to be done once

# preprocessing and sampling
for i, dataset in enumerate(datasets):
    dataset_raw = os.path.join(path_raw, dataset)
    dataset_processed = os.path.join(path_processed, dataset)
    # To avoid name conflicts every dataset get an index
    preprocess(dataset_raw, dataset_processed, i, ratio=(0, 0, 1))
    create_samples(dataset_processed, 'test')

processed_datasets = [os.path.join(path_processed, data) for data in datasets]
model = tf.keras.models.load_model('models/trained_model')

# evaluation
for i, test_set in enumerate(processed_datasets):
    test_set_name = test_set.split('/')[-1]
    test_ds, _, _ = make_dataset('test', [test_set_name])
    scores = model.predict(test_ds)
    predictions = scores.argmax(axis=1)

    test_samples = os.path.join(test_set, 'samples_test.csv')
    samples_df = pd.read_csv(test_samples)
    samples_df['prediction'] = predictions
    samples_df.to_csv(os.path.join(test_set, 'predictions.csv'))
    samples_df.to_csv(f'report/preds_{test_set_name}.csv')

    gt = samples_df['label']
    labels = [i[0] for i in read_label_map(os.path.join(path_processed, data['label_map'])).items() if
              i[1] in predictions or i[1] in gt.values]
    create_confusion_matrix(gt, predictions, labels, f'confusion_matrix_{test_set_name}')

# Dummy file. See https://github.com/iterative/dvc/issues/8881. May be resolved in future
f = open("temp/dummy_evaluate_test.txt", "w")
f.write("Evaluation Test set complete")
f.close()
