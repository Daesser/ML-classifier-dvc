import os
import yaml
from glob import glob
import tensorflow as tf
from imgaug import augmenters as iaa
import cv2
from scipy.interpolate import UnivariateSpline
import numpy as np
import random
import tensorflow_addons as tfa

"""
This module provides functions to create tf.Datasets for training and testing. Using tf.Datasets is more performant.
For more information refers to:  https://www.tensorflow.org/guide/data_performance.
This module may be too complex and could be reworked.
"""

params = yaml.safe_load(open('params.yaml'))['prepare']
test_specs = yaml.safe_load(open('params.yaml'))['test']
data = yaml.safe_load(open('params.yaml'))['data']
datasets = data["datasets"]  # Array of dataset names
path_processed = data["dataset_processed"]
image_width = params['width']
image_height = params['height']
buffer_size = params['buffer_size']


def make_dataset(split='train', path=datasets, gamma=False, blur=False,
                 tone=None, brightness=False, hue=False, downsample=False):
    """This function will create a dataset of Dataset for training or testing from the samples saved in the
    samples.csv file using the tf.data API.
    For more information refers to:  https://www.tensorflow.org/guide/data_performance

    Parameters:
    split (string): Purpose of the dataset (train,val or test).
    path (array): Path to a processed datasets
    gamma (boolean): Adjust gamma distortion if True for robust testing
    blur (boolean): Apply blur if True for robust testing
    tone (string): Apply toning if set for robust testing
    brightness (boolean): Adjust brightness if True for robust testing
    hue (boolean): Adjust hue if True for robust testing
    downsample (boolean): Down sampling and up sampling if True for robust testing
    Returns:
    dataset (object): tf.Dataset object
    """

    def parse_image(filename):
        """This function parse the image and applies simple distortions if split is indicated as 'test'.

        Parameters:
        filename (string): Path of filename to read from
        Returns:
        image (object): Tensor of shape [new_height, new_width, channels].
        """
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        if split == 'test':
            if downsample:
                image = tf.image.resize(image, [int(image_width / 2), int(image_height / 2)])
                image = tf.image.resize(image, [image_width, image_height])
            if gamma:
                image = tf.image.adjust_gamma(image, test_specs['gamma'])
            if brightness:
                image = tf.image.adjust_brightness(image, test_specs['delta'])
            if hue:
                image = tf.image.adjust_hue(image, test_specs['hue'])
            if blur:
                image = tfa.image.gaussian_filter2d(image, filter_shape=(7, 7))

        image = tf.image.resize(image, [image_width, image_height])

        return image

    def get_num_images():
        """This function returns the number of images.

        Parameters:
        Returns:
        num_images (int): Number of images used for training
        """
        num_images = 0
        for dataset in datasets:
            dataset_path = os.path.join(path_processed, dataset)
            images = glob(dataset_path + f'/*/{split}/*/*')
            num_images = num_images + len(images)
        return num_images

    def get_num_classes():
        """This function returns the number classes by taking maximimum classes among the datasets
        used for training.

        Parameters:
        Returns:
        num_images (int): Number of classes
        """
        num_classes = 0
        for dataset in datasets:
            front_path = os.path.join(path_processed, dataset, 'front', split)
            classes = os.listdir(front_path)
            if len(classes) > num_classes:
                max_classes = classes
                num_classes = len(classes)

        return max_classes

    def normalize(x, y, z):
        return x / 255., y / 255., z / 255.

    def spread_lookup_table(x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))

    def add_tone(image, tone=None):
        """This function add a tone to the image.

        Parameters:
        image (array): np.array of shape [new_height, new_width, channels].
        tone (string): 'red', 'blue' or None supported.
        Returns:
        image (array): np.array of shape [new_height, new_width, channels].
        """
        increase_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 80, 160, 256])
        decrease_lookup_table = spread_lookup_table([0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel = cv2.split(image)
        if tone == 'red':
            red_channel = cv2.LUT(red_channel.astype(np.uint8), increase_lookup_table).astype(np.uint8)
            blue_channel = cv2.LUT(blue_channel.astype(np.uint8), decrease_lookup_table).astype(np.uint8)
        if tone == 'blue':
            red_channel = cv2.LUT(red_channel.astype(np.uint8), decrease_lookup_table).astype(np.uint8)
            blue_channel = cv2.LUT(blue_channel.astype(np.uint8), increase_lookup_table).astype(np.uint8)
        green_channel = green_channel.astype(red_channel.dtype)
        return cv2.merge((red_channel, green_channel, blue_channel))

    def add_tone_randomly(image, tone=None):
        """This function add a tone to the image randomly in 1/3 of the cases.

        Parameters:
        image (array): np.array of shape [new_height, new_width, channels].
        tone (string): 'red', 'blue' or None supported.
        Returns:
        image (array): np.array of shape [new_height, new_width, channels].
        """
        rand_number = random.randint(0, 9)
        if rand_number < 3:
            return add_tone(image, tone)
        else:
            return image

    def random_toning(front, back, side, tone):

        return add_tone_randomly(front.numpy(), tone), \
               add_tone_randomly(back.numpy(), tone), \
               add_tone_randomly(side.numpy(), tone)

    def augment(front, back, side):
        """This function applies random augmentation to the images.
        For more information about the augmentation technique see this paper: https://arxiv.org/abs/1909.13719.

        Parameters:
        front (array): Tensor of shape [new_height, new_width, channels] of the front image.
        back (string): Tensor of shape [new_height, new_width, channels] of the back image.
        side (string): Tensor of shape [new_height, new_width, channels] of the side image.
        Returns:
        images (tuple): np.array of shape [new_height, new_width, channels] of the three images.
        """
        rand_aug = iaa.RandAugment(n=params['randaugment_n'], m=params['randaugment_m'])
        # Input to `augment()` is a TensorFlow tensor which is not supported by `imgaug`. This is why we first
        # convert it to its `numpy` variant.
        front = tf.cast(front, tf.uint8)
        back = tf.cast(back, tf.uint8)
        side = tf.cast(side, tf.uint8)
        front = rand_aug(images=front.numpy())
        back = rand_aug(images=back.numpy())
        side = rand_aug(images=side.numpy())

        return front, back, side

    def prepare_images(front, back, side):
        front_image = parse_image(front)
        back_image = parse_image(back)
        side_image = parse_image(side)
        return front_image, back_image, side_image

    def configure_for_performance(ds):
        if split == 'train':
            ds = ds.shuffle(buffer_size=buffer_size)
            ds = ds.repeat()
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def load_csv_dataset():
        """This function creates a tf.Dataset from all datasets by loading the samples saved in samples.csv.
        Returns:
        dataset (object): tf.Dataset object.
        """
        for i, datas in enumerate(path):
            samples_csv = os.path.join(path_processed, datas, f'samples_{split}.csv')
            if i == 0:

                dataset = tf.data.experimental.CsvDataset(
                    samples_csv,
                    [tf.string,
                     tf.string,
                     tf.string,
                     ],
                    select_cols=[0, 1, 2],
                    header=True  # Only parse last three columns
                )

                labels = tf.data.experimental.CsvDataset(
                    samples_csv,
                    [tf.int32
                     ],
                    select_cols=[9],  # Column 9 corresponds to label
                    header=True
                )
            else:
                temp_dataset = tf.data.experimental.CsvDataset(
                    samples_csv,
                    [tf.string,
                     tf.string,
                     tf.string,
                     ],
                    select_cols=[0, 1, 2],
                    header=True
                )
                temp_labels = tf.data.experimental.CsvDataset(
                    samples_csv,
                    [tf.int32
                     ],
                    select_cols=[9],  # Column 9 corresponds to label
                    header=True
                )
                dataset = dataset.concatenate(temp_dataset)
                labels = labels.concatenate(temp_labels)

        return dataset, labels

    dataset, labels = load_csv_dataset()
    dataset = dataset.map(lambda x, y, z: prepare_images(x, y, z),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if split == 'test':
        if tone:
            dataset = dataset.map(
                lambda x, y, z: (tf.py_function(random_toning, [x, y, z, tone], [tf.float32, tf.float32, tf.float32])),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(params['batch_size'])
    labels = labels.batch(params['batch_size'])

    if split == 'train':
        dataset = dataset.map(
            lambda x, y, z: (tf.py_function(augment, [x, y, z], [tf.float32, tf.float32, tf.float32])),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda x, y, z: normalize(x, y, z))

    dataset = tf.data.Dataset.zip((dataset, labels))
    dataset = configure_for_performance(dataset)
    images = get_num_images()
    classes = get_num_classes()

    return dataset, images, classes


def get_dataset():
    train_ds, num_images, classes = make_dataset('train')
    val_ds, _, _ = make_dataset('val')

    return train_ds, val_ds, classes, num_images
