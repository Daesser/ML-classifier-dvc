import os
import glob
import yaml
from pathlib import Path
from src.expceptions.InvalidImageError import InvalidImageError

"""
This script is used to validate the raw data. It is assume the following folder structure
dataset1/type/view/flash/distance/image.jpg
dataset2/type/view/flash/distance/image.jpg
...
"""

data = yaml.safe_load(open('params.yaml'))['data']
path_raw = data["dataset_raw"]  # Root directory of raw data
datasets = data["datasets"]  # Array of dataset names


def is_image(path):
    """
    Check if the filepath contains an image
    """
    image = open(path, 'rb').read(8)  # First 8 bytes needed to determine type of image
    # check if file is JPG or JPEG
    if image[:3] == b'\xff\xd8\xff':
        return True
    # check if file is PNG
    if image[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        return True
    # check if file is GIF
    if image[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        return True
    return False


def validate_images(path=path_raw, delete=False):
    """Check if all files are images assuming the folder structure:

    Parameters:
    path (string): Path to the folder where the datasets are saved
    delete (boolean): Files of format other than jpg,jpeg, png, gif will be deleted if True
    """
    raw_images = glob.glob(f'{path}/*/*/*/*/*')
    for image in raw_images:
        if not is_image(image):
            if delete:
                os.remove(image)
            else:
                raise InvalidImageError(image)


if __name__ == '__main__':
    for dataset in datasets:
        dataset = os.path.join(path_raw, dataset)
        validate_images(dataset)

    # Dummy file. See https://github.com/iterative/dvc/issues/8881. May be resolved in future
    Path('temp').mkdir(parents=True, exist_ok=True)
    f = open("temp/dummy_validation.txt", "w")
    f.write("Validation complete")
    f.close()
