import yaml
import math
from src.data.make_dataset import get_dataset
from src.models.model import get_model
import tensorflow as tf
from dvclive import Live
import shutil

"""
This script starts the training process and saves the models.
The models will be saved under models. DVCLive is used for logging some metrics.
For more information refers to:  https://dvc.org/doc/dvclive.
"""

filename = 'reports/metrics/training.csv'
params = yaml.safe_load(open('params.yaml'))['train']
data = yaml.safe_load(open('params.yaml'))['data']

live = Live('reports/metrics/metrics')
checkpoint_filepath = 'models/best'


def scheduler(epoch, lr):
    if epoch < 55:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=False)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    save_best_only=True)


def fit_model():
    train_dataset, val_dataset, classes, num_images = get_dataset()
    model = get_model('ef', len(classes))

    history = model.fit(train_dataset, epochs=params['epochs'],
                        callbacks=[history_logger, model_checkpoint_callback, scheduler_callback],
                        validation_data=val_dataset,
                        steps_per_epoch=math.ceil(num_images))

    # Use live.lof_metric() with the newer version
    live.log("train_acc", history.history['sparse_categorical_accuracy'][-1])
    live.log("val_acc", history.history['val_sparse_categorical_accuracy'][-1])
    live.log("train_loss", history.history['loss'][-1])
    live.log("val_loss", history.history['val_loss'][-1])
    model.save('models/trained_model')

    # dvc studio need two different file names, change later
    shutil.copyfile(filename, 'reports/metrics/training_val.csv')


if __name__ == '__main__':
    fit_model()

    # Dummy file. See https://github.com/iterative/dvc/issues/8881. May be resolved in future
    f = open("temp/dummy_train.txt", "w")
    f.write("Training complete")
    f.close()
