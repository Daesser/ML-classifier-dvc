import yaml
import random
import csv
import os
import glob as glob
import numpy as np
from src.data.preprocess import read_label_map

"""
This script creates samples of three pictures (front, back, side) for training and save the created samples
in a csv file. take the images in the raw folder and preprocess them (see chapter 3 in documentation).
Sampling requires searching the tags.csv file for images with the same tags if indicated which is slow. If the tags
information can be derived from the filename (e.g. FNT-10EIWD_far_flash.jpg) sampling  is much faster.
Functions ending with 'name' get the tags from the filename and the csv file doesn't need to be opened.
This script creates samples_train.csv, samples_val.csv, samples_test.csv in the data/processed/data_name folder.
"""

params = yaml.safe_load(open('params.yaml'))['sample']
data = yaml.safe_load(open('params.yaml'))['data']

path_processed = data["dataset_processed"]  # Root directory of processed data
datasets = data["datasets"]  # Array of dataset names
label_map_file = data['label_map']  # Label map file


def get_tags(path, filepath):
    """This function returns the tags of an image as a dict by looking them up in the tags.csv file.

    Parameters:
    path (string): Path to a processed dataset
    filepath (string): Path to the image
    Returns:
    tags (dict): Dict with the tags of the image
    """
    filename = filepath.split('/')[-1]
    split = filepath.split('/')[-3]  # train, val or test
    csv_path = os.path.join(path, f'tags_{split}.csv')
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        tags = next(filter(lambda file: file['filename'] == filename, reader), None)
        tags['split'] = split
    return tags


def get_tags_name(filepath):
    """This function returns the tags of an image as a dict derived by the filename.

    Parameters:
    filepath (string): Path to the image
    Returns:
    tags (dict): Dict with the tags of the image
    """
    filename = filepath.split('/')[-1]
    specs = filepath.split('/')
    name = filename.split('_')
    return {'filename': specs[-1],
            'distance': name[-3],
            'light': name[-4],
            'view': specs[-4],
            'class': specs[-2],
            'split': specs[-3]}


def get_back_list(path, tags, same_distance, same_flash):
    """This function returns a filtered list of images with the same tags as indicated in the tags parameter by
    looking them up in the tags.csv file. Images with same distance and flash are returned when set to True.

    Parameters:
    path (string): Path to the processed dataset
    tags (string): Tags of the (front) image
    same_distance (boolean): Whether only images with the same distance should be returned
    same_flash (boolean): Whether only images with the same flash should be returned
    Returns:
    back_list (array): Filtered array containing paths of images
    """
    split = tags['split']
    class_path = os.path.join(path, 'back', split, tags["class"])

    csv_path = os.path.join(path, f'tags_{split}.csv')
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        if same_distance and same_flash:
            back_list = [os.path.join(class_path, file['filename']) for file in reader
                         if file['distance'] == tags['distance']
                         and file['light'] == tags['light']
                         and file['view'] == 'back'
                         and file['class'] == tags['class']]
        if same_distance and not same_flash:
            back_list = [os.path.join(class_path, file['filename']) for file in reader
                         if file['distance'] == tags['distance']
                         and file['view'] == 'back'
                         and file['class'] == tags['class']]
        if not same_distance and not same_flash:
            back_list = [os.path.join(class_path, file['filename']) for file in reader
                         if file['view'] == 'back'
                         and file['class'] == tags['class']]
        if not same_distance and same_flash:
            back_list = [os.path.join(class_path, file['filename']) for file in reader
                         if file['view'] == 'back'
                         and file['light'] == tags['light']
                         and file['class'] == tags['class']]

    return back_list


def get_back_list_name(path, tags, same_distance, same_flash):
    """This function returns a filtered list of images with the same tags as indicated in the tags parameter by
    comparing the names of the files. Images with same distance and flash are returned when set to True.

    Parameters:
    path (string): Path to the 'back' images
    tags (string): Tags of the (front) image
    same_distance (boolean): Whether only images with the same distance should be returned
    same_flash (boolean): Whether only images with the same flash should be returned
    Returns:
    back_list (array): Filtered array containing paths of images
    """
    paths = glob.glob(path + '/*')
    if same_distance and same_flash:
        back_list = [path for path in paths
                     if path.split('_')[-3] == tags['distance']
                     and path.split('_')[-4] == tags['light']]
    if same_distance and not same_flash:
        back_list = [path for path in paths
                     if path.split('_')[-3] == tags['distance']]
    if not same_distance and not same_flash:
        back_list = paths
    if not same_distance and same_flash:
        back_list = [path for path in paths
                     if path.split('_')[-4] == tags['light']]

    return back_list


def get_side_list(path, tags, same_flash):
    """This function returns a filtered list of images with the same tags as indicated in the tags parameter by
    looking them up in the tags.csv file. Images with same flash are returned when set to True.

    Parameters:
    path (string): Path to the processed dataset
    tags (string): Tags of the (front) image
    same_flash (boolean): Whether only images with the same flash should be returned
    Returns:
    side_list (array): Filtered array containing paths of images
    """
    split = tags['split']
    csv_path = os.path.join(path, f'tags_{split}.csv')
    class_path = os.path.join(path, 'side', split, tags["class"])

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if same_flash:
            side_list = [os.path.join(class_path, file['filename']) for file in reader
                         if file['light'] == tags['light']
                         and file['view'] == 'side'
                         and file['class'] == tags['class']]
        else:
            side_list = [os.path.join(class_path, file['filename']) for file in reader
                         if file['view'] == 'side'
                         and file['class'] == tags['class']]

    return side_list


def get_side_list_name(path, tags, same_flash):
    """This function returns a filtered list of images with the same tags as indicated in the tags parameter by
    comparing the names of the files. Images with same flash are returned when set to True.

    Parameters:
    path (string): Path to the 'side' images
    tags (string): Tags of the (front) image
    same_flash (boolean): Whether only images with the same flash should be returned
    Returns:
    back_list (array): Filtered array containing paths of images
    """
    paths = glob.glob(path + '/*')
    if same_flash:
        side_list = [path for path in paths
                     if path.split('_')[-4] == tags['light']]
        return side_list

    return paths


def sample_image(path, filepath, view, same_distance, same_flash):
    """This function sample an image.

    Parameters:
    path (string): Path to a processed dataset
    filepath (string): Path to the image
    view (string): The view of the sampled image
    same_distance (boolean): Whether an image with the same distance should be returned
    same_flash (boolean): Whether an image with the same flash should be returned
    Returns:
    image (tuple): sampled image, tags of sampled image
    """
    # tags = get_specs(path, filepath)
    tags = get_tags_name(filepath)
    split = tags['split']
    if view == 'back':
        class_path = os.path.join(path, 'back', split, tags["class"])
        files = get_back_list_name(class_path, tags, same_distance, same_flash)
        # files = get_back_list(path, tags, same_distance, same_flash)
    else:
        class_path = os.path.join(path, 'side', split, tags["class"])
        files = get_side_list_name(class_path, tags, same_flash)
        # files = get_side_list(path, tags, same_flash)
    rand_int = random.randint(0, len(files) - 1)
    image = files[rand_int]

    return image, get_tags_name(image)


def sample_images(path, front_images, view, same_distance, same_flash):
    """This function returns a list of sampled images.

    Parameters:
    path (string): Path to a processed dataset
    front_images (array): List of front image paths
    view (string): The view of the sampled image
    same_distance (boolean): Whether an image with the same distance should be returned
    same_flash (boolean): Whether an image with the same flash should be returned
    Returns:
    image (tuple): array of sampled image, tags of sampled images
    """
    images = [sample_image(path, front_image, view, same_distance, same_flash)[0] for front_image in front_images]
    specs = [sample_image(path, front_image, view, same_distance, same_flash)[1] for front_image in front_images]
    return images, specs


def create_samples(path, split, side=True):
    """This function creates samples and save them into a csv file (samples.csv).

    Parameters:
    path (string): Path to a processed dataset
    split (string): Split: train, val or test
    side (boolean): Whether also sampling 'side' images
    """
    repeat = params[f'repeat_{split}']
    front_path = os.path.join(path, 'front', split)
    front_images = glob.glob(front_path + '/*/*')
    label_map = read_label_map(os.path.join(path_processed, label_map_file))
    front_images = np.repeat(front_images, repeat)
    labels = [label_map[name.split('/')[-2]] for name in front_images]

    # Use filename to get the tags. If the tags are not in filename use get_specs (much slower).
    # front_lights = [get_specs(path, front_image)['light'] for front_image in front_images]
    # front_dist = [get_specs(path, front_image)['distance'] for front_image in front_images]

    front_lights = [name.split('_')[-4] for name in front_images]  # derived from file name
    front_dist = [name.split('_')[-3] for name in front_images]  # derived from file name

    back_images, back_specs = sample_images(path, front_images, view='back', same_distance=False, same_flash=False)
    back_lights = [item['light'] for item in back_specs]
    back_dist = [item['distance'] for item in back_specs]

    if side:
        # sample side images
        side_images, side_specs = sample_images(path, front_images, view='side', same_distance=False, same_flash=False)
        side_lights = [item['light'] for item in side_specs]
        side_dist = [item['distance'] for item in back_specs]
        images = np.stack([front_images, back_images, side_images,
                           front_dist, back_dist, side_dist,
                           front_lights, back_lights, side_lights,
                           labels], axis=1)
    else:
        images = np.stack([front_images, back_images, ['None'] * len(front_images),
                           front_dist, back_dist, ['None'] * len(front_images),
                           front_lights, back_lights, ['None'] * len(front_images),
                           labels], axis=1)

    output_csv = os.path.join(path, f'samples_{split}.csv')
    with open(output_csv, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['front', 'back', 'side',
                         'f_dis', 'b_dis', 's_dis',
                         'f_flash', 'b_flash', 's_flash',
                         'label'])

        writer.writerows(images)


def sample_from_custom_dataset(path):
    create_samples(path, 'test', side=True)


if __name__ == '__main__':

    for dataset in datasets:
        out_base_path = os.path.join(path_processed, dataset)
        create_samples(out_base_path, 'train')
        print('Train Sampling finished')
        create_samples(out_base_path, 'val')
        print('Val Sampling finished')
        create_samples(out_base_path, 'test')
        print('Test Sampling finished')

    # Dummy file. See https://github.com/iterative/dvc/issues/8881. May be resolved in future
    f = open("temp/dummy_sample.txt", "w")
    f.write("Sampling complete")
    f.close()
