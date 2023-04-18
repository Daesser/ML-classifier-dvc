import os
import splitfolders
import yaml
import shutil
from pathlib import Path
import glob
import csv
import cv2

"""
This script takes the images in the raw folder and preprocess them. The processed dataset is saved in the
data/processed folder (see chapter 3 in documentation). The interim folder is used to temporary store the images
and will be deleted at the end as the splitfolders library requires a certain folder structure.
This script creates tags.csv, tags_train.csv, tags_val.csv, tags_test.csv in the data/processed/data_name folder.
"""

params = yaml.safe_load(open('params.yaml'))['preprocess']
data = yaml.safe_load(open('params.yaml'))['data']
path_raw = data["dataset_raw"]  # Root directory of raw data
path_processed = data["dataset_processed"]  # Root directory of processed data
datasets = data["datasets"]  # Array of dataset names
label_map_file = data['label_map']  # Label map file
views = ['front', 'back', 'side']
specs = ['filename', 'distance', 'light', 'view', 'class']
temporary_folder = 'data/interim/Belt'


def create_label_map(path_dataset):
    """This function creates a label map based on the different names of the folders containing the images.
       You may change to a more standardized format in the future.
    Parameters:
    path_dataset(string): Path to a raw dataset
    """
    Path(path_processed).mkdir(parents=True, exist_ok=True)
    classes = os.listdir(path_dataset)
    filename = os.path.join(path_processed, label_map_file)
    with open(filename, 'w') as the_file:
        for belt, index in enumerate(classes):
            the_file.write('item\n')
            the_file.write('{\n')
            the_file.write('id :{}'.format(int(belt)))
            the_file.write('\n')
            the_file.write("name :'{0}'".format(str(index)))
            the_file.write('\n')
            the_file.write('}\n')


def read_label_map(label_map_path):
    """This function read a label map created by the create_label_map function.

    Parameters:
    path_dataset (string): Path to the label map file

    Returns:
    items (dict): Dict with type name and number
    """
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items


def split_dataset(path_out, ratio=(params['train_split'], params['val_split'], params['test_split'])):
    """This function splits the dataset in train, val and test dataset.

    Parameters:
    path_out (string): Path to the output folder
    ratio (tuple): Split ratio (train,val,test)
    """
    for view in views:
        view_path = os.path.join(temporary_folder, view)
        out_path = os.path.join(path_out, view)
        splitfolders.ratio(view_path,
                           output=out_path,
                           seed=params['seed'],
                           ratio=(float(ratio[0]), float(ratio[1]), float(ratio[2])),
                           group_prefix=None,
                           move=False)
    print('Split complete')


def create_split_csv(path, csv_in, csv_out, split='val'):
    """This function creates a csv file with tags after the split.

    Parameters:
    path (string): Path to a processed dataset
    csv_in (string): Name of csv file created with the function save_tags
    csv_out (string): Name of the csv file
    split (string): Split the csv is created for
    """
    files = glob.glob(path + f'/*/{split}/*/*')

    filenames = [x.split('/')[-1] for x in files]
    csv_file_in = os.path.join(path, csv_in)
    csv_file_out = os.path.join(path, csv_out)
    with open(csv_file_in, 'r') as file_in:
        with open(csv_file_out, 'w') as file_out:
            csvreader = csv.reader(file_in)
            writer = csv.writer(file_out)
            writer.writerow(specs)

            for row in csvreader:
                if row[0] in filenames:  # row[0] = filename
                    writer.writerow(row)


def save_tags(path, path_out, name, dataset_index, classes):
    """This function saves the tags of the images in a csv file.

    Parameters:
    path (string): Path to a raw dataset
    path_out (string): Path to the processed dataset
    name (string): Name of the csv file (tags.csv)
    dataset_index (int): Index of the dataset
    classes (array): Belt type names
    """
    csv_file = os.path.join(path_out, name)
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path_out).mkdir(parents=True, exist_ok=True)
    with open(csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(specs)
        for belt_type in classes:
            for view in views:
                images = glob.glob(path + f'/{belt_type}/{view}/*/*/*')
                for index, image in enumerate(images):
                    image_spec = image.split("/")[3:-1]  # exclude first 3 (path to dataset)
                    filename = f'{image_spec[0]}_{image_spec[1]}_{image_spec[2]}_' \
                               f'{image_spec[3]}_{index}_{dataset_index}.jpg'
                    image_spec.append(filename)
                    writer.writerow(image_spec[::-1])  # reverse order

    print('Tags saved')


def merge(path, dataset_index, classes):
    """This function will merge all images together. The folder structure changes as follows:
    dataset/type/view/light/distance->dataset/view/type

    Parameters:
    path_in (string): Path to a raw dataset
    dataset_index (int): Index of the dataset
    classes (array): Belt type names
    """
    Path(path_processed).mkdir(parents=True, exist_ok=True)
    for belt_type in classes:
        for view in views:
            class_path = os.path.join(path, belt_type, view)
            temp_folder = os.path.join(temporary_folder, view, belt_type)
            Path(class_path).mkdir(parents=True, exist_ok=True)
            Path(temp_folder).mkdir(parents=True, exist_ok=True)
            images = glob.glob(path + f'/{belt_type}/{view}/*/*/*')
            for index, image in enumerate(images):
                image_spec = image.split("/")[3:-1]  # exclude first 3 (path to dataset)
                filename = f'{image_spec[0]}_{image_spec[1]}_{image_spec[2]}_' \
                           f'{image_spec[3]}_{index}_{dataset_index}.jpg'
                shutil.copyfile(image, os.path.join(temp_folder, filename))


def rotate():
    """This function rotates the images in the temporary folder so it has the same alignment.
    """
    paths = glob.glob(temporary_folder + '/*/*/*')
    for path in paths:
        try:
            image = cv2.imread(path)
            if image.shape[0] > image.shape[1]:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(path, image)
        except OSError:
            print(path)


def preprocess(raw, processed, index, ratio=(params['train_split'], params['val_split'], params['test_split'])):
    """This function preprocesses a raw dataset by splitting and make the data ready for training
    as shown in Chapter 3.

    Parameters:
    raw (string): Path to a raw dataset
    processed (string): Path to the processed dataset
    index (int): Index of the dataset
    ratio (tuple): Split ratio (train,val,test)
    """
    classes = [item for item in read_label_map(os.path.join(path_processed, label_map_file))]
    save_tags(raw, processed, 'tags.csv', index, classes)
    merge(raw, index, classes)
    rotate()
    split_dataset(processed, ratio)
    create_split_csv(processed, 'tags.csv', 'tags_val.csv', 'val')
    create_split_csv(processed, 'tags.csv', 'tags_train.csv', 'train')
    create_split_csv(processed, 'tags.csv', 'tags_test.csv', 'test')
    shutil.rmtree(temporary_folder)


if __name__ == '__main__':

    # Create label map
    """
    first_dataset = datasets[0]  # We use the first dataset specified for creating the label map
    create_label_map(os.path.join(path_raw, first_dataset))
    """

    # Save the tags of the images in a csv file before merging
    for i, dataset in enumerate(datasets):
        dataset_raw = os.path.join(path_raw, dataset)
        dataset_processed = os.path.join(path_processed, dataset)
        # To avoid name conflicts every dataset get an index
        preprocess(dataset_raw, dataset_processed, i)

    # Dummy file. See https://github.com/iterative/dvc/issues/8881. May be resolved in future
    f = open("temp/dummy_preprocess.txt", "w")
    f.write("Preprocessing complete")
    f.close()
