import yaml
from dvclive import Live
import tensorflow as tf
from src.data.make_dataset import get_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.data.preprocess import read_label_map

"""
This script make predictions for the validation set and save them into a csv file.
It also creates confusion matrices and classification reports.
"""

data = yaml.safe_load(open('params.yaml'))['data']
path_processed = data["dataset_processed"]  # Root directory of processed data
datasets = data["datasets"]  # Array of dataset names
model = tf.keras.models.load_model('models/trained_model')


def create_confusion_matrix(gt, preds, labels, name):
    """This function creates confusion_matrix.json and  conf.png.

    Parameters:
    gt(array): Array of true labels
    preds(array): Array of predictions
    labels(array): Array of labels
    name(string): File name to be saved
    """
    cf_matrix = confusion_matrix(gt.values, preds)
    df_cm = pd.DataFrame(cf_matrix, index=labels,
                         columns=labels)
    plt.figure(figsize=(50, 45))
    s = sns.heatmap(df_cm, annot=True)
    s.set(xlabel='Predicted', ylabel='Actual')
    plt.savefig(f'report/figures/{name}.png', dpi=400)


def create_classification_report(labels, preds):
    """This function creates a classification report and save it as svm_conf.png

    Parameters:
    labels(array): Array of true labels
    """
    clf_report = classification_report(labels, preds, target_names=classes, output_dict=True)
    svm = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    figure = svm.get_figure()
    figure.savefig('report/figures/svm_conf.png', dpi=400)


if __name__ == '__main__':
    _, val_ds, classes, _ = get_dataset()
    scores = model.predict(val_ds)
    predictions = scores.argmax(axis=1)

    all_datasets = [os.path.join(path_processed, dataset, 'samples_val.csv') for dataset in datasets]
    df_datasets = (pd.read_csv(f) for f in all_datasets)
    samples_df = pd.concat(df_datasets, ignore_index=True)
    samples_df['prediction'] = predictions
    gt = samples_df['label']
    # save the predictions and scores
    samples_df.to_csv('report/predictions_val.csv')
    pd.DataFrame(scores).to_csv('report/scores_val.csv')
    labels = [i for i in read_label_map(os.path.join(path_processed, data['label_map']))]
    # creates plots
    live = Live('reports')
    # live.log_plot("roc", y_true, y_score)
    live.log_plot('confusion_matrix', gt, predictions)
    create_confusion_matrix(gt, predictions, labels, 'confusion_matrix_val')
    create_classification_report(gt, predictions)

    # Dummy file. See https://github.com/iterative/dvc/issues/8881. May be resolved in future
    f = open("temp/dummy_evaluate.txt", "w")
    f.write("Evaluation complete")
    f.close()
