import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import utils_graph
import utils_io

S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/eigenfaces/'
FILE_ZIP = 'yalefaces_cropped.zip'
FILE_TAR = 'yalefaces_uncropped.tar'
FOLDER_CROPPED = 'CroppedYale'
FOLDER_UNCROPPED = 'yalefaces'
DATA_FOLDER = 'data'
PLOTS_FOLDER = 'plots'
DATA_FILE = 'data.npz'
UNCROPPED_WIDTH = 320
UNCROPPED_HEIGHT = 243
CROPPED_WIDTH = 168
CROPPED_HEIGHT = 192


def preprocess_data() -> None:
    """Downloads and preprocesses data.
    """
    print('Downloading and preprocessing data.')

    (_, downloaded1) = utils_io.download_remote_data_file(DATA_FOLDER, S3_URL+FILE_ZIP)
    if downloaded1:
        utils_io.unpack_zip_file(DATA_FOLDER, FILE_ZIP)
        # Convert .pgm images to .png so we can easily visualize them.
        path_cropped = os.path.join(DATA_FOLDER, FOLDER_CROPPED)
        utils_io.convert_images(path_cropped, '.png')

    (_, downloaded2) = utils_io.download_remote_data_file(DATA_FOLDER, S3_URL+FILE_TAR)
    if downloaded2:
        utils_io.unpack_tar_file(DATA_FOLDER, FILE_TAR)
        # Add a .gif extension to uncropped images if it hasn't been added
        # already.
        path_uncropped = os.path.join(DATA_FOLDER, FOLDER_UNCROPPED)
        utils_io.append_to_all_files(path_uncropped, '.gif')

    print('Done downloading and preprocessing data.')


def _recursive_read_cropped(folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function that reads the cropped data and merges it into a single 
    matrix, recursively.
    """
    # 32256 is the width*height of each image we'll read.
    xc = np.empty((CROPPED_WIDTH*CROPPED_HEIGHT,0))
    xc_labels = np.empty((0,))
    items = os.listdir(folder)
    for item in items:
        item = os.path.join(folder, item)
        if os.path.isdir(item):
            (new_xc, new_label) = _recursive_read_cropped(item)
            xc = np.append(xc, new_xc, 1)
            xc_labels = np.append(xc_labels, new_label)
        elif os.path.isfile(item) and item.endswith('.pgm'):
            try:
                image = Image.open(item)
                image = image.getdata()
                image = np.asarray(image)
                image = np.reshape(image, (image.shape[0], 1))
                xc = np.append(xc, image, 1)
                parent_dir = os.path.dirname(item)
                label = parent_dir[-3:]
                xc_labels = np.append(xc_labels, label)
            except IOError:
                print(f'Cannot open file {item}.')
    return (xc, xc_labels)


def _read_cropped() -> Tuple[np.ndarray, np.ndarray]:
    """Reads the cropped data and labels.
    """
    print('Reading cropped images.')
    path_cropped = os.path.join(DATA_FOLDER, FOLDER_CROPPED)
    return _recursive_read_cropped(path_cropped)
    print('Done reading cropped images.')


def _read_uncropped() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads the uncropped data and labels.
    """
    print('Reading uncropped images.')
    # 77760 is the width*height of each image we'll read.
    xu = np.empty((UNCROPPED_WIDTH*UNCROPPED_HEIGHT,0))
    # This will contain the subject number of the image.
    xu_labels_1 = np.empty((0,))
    # This will contain a tag such as 'happy', or 'glasses'.
    xu_labels_2 = np.empty((0,))
    path_uncropped = os.path.join(DATA_FOLDER, FOLDER_UNCROPPED)
    items = os.listdir(path_uncropped)
    for item in items:
        item = os.path.join(path_uncropped, item)
        if os.path.isfile(item):
            try:
                image = Image.open(item)
                image = image.getdata()
                image = np.asarray(image)
                image = np.reshape(image, (image.shape[0], 1))
                xu = np.append(xu, image, 1)
                item_sections = item.split('.')
                xu_labels_1 = np.append(xu_labels_1, item_sections[0][-2:])
                xu_labels_2 = np.append(xu_labels_2, item_sections[1])
            except IOError:
                print(f'Cannot open file {item}.')
    print('Done reading uncropped images.')
    return (xu, xu_labels_1, xu_labels_2)


def read_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
    np.ndarray]:
    """Reads the cropped and uncropped data and labels.
    """
    data_file = os.path.join(DATA_FOLDER, DATA_FILE)
    data = ()
    if os.path.exists(data_file):
        data_dict = np.load(data_file) 
        data = (data_dict[key] for key in data_dict)
    else:
        (xc, xc_labels) = _read_cropped()
        (xu, xu_labels_1, xu_labels_2) = _read_uncropped()
        data = (xc, xc_labels, xu, xu_labels_1, xu_labels_2)
        np.savez(data_file, *data)
    return data


def _save_image(array: np.ndarray, filename: str, image_type: str) -> Image:
    """Saves an image, given a flat array of double values.
    """
    # We reshape the array according to the image type.
    if image_type == 'cropped':
        image_shape = (CROPPED_HEIGHT, CROPPED_WIDTH)
    elif image_type == 'uncropped':
        image_shape = (UNCROPPED_HEIGHT, UNCROPPED_WIDTH)
    matrix = np.reshape(array, image_shape)

    # We convert the values in the image to go from 0 to 255 and be ints.
    matrix -= matrix.min()
    matrix *= 255/matrix.max()
    matrix = matrix.astype('uint8')
    image = Image.fromarray(matrix, mode='L')
    image.save(os.path.join(PLOTS_FOLDER, filename))


def svd_analysis(x: np.ndarray, plots_dir: str, image_type: str) -> None:
    """Decomposes matrix x using SVD and plots the singular values and modes.
    """
    print(f'Analyzing SVD of {image_type} images.')
    (u, s, vh) =  np.linalg.svd(x, full_matrices=False)

    # Plot the normalized singular values. (Just the first 100 singular values.)
    normalized_s = s / np.sum(s)
    normalized_s = normalized_s[:100]
    utils_graph.graph_2d_markers(
        np.asarray(range(1, len(normalized_s)+1)),
        np.log(normalized_s), 'Mode', 'Log of normalized singular value',
        f'Singular values for {image_type} images', 
        plots_dir, 
        f'singular_values_{image_type}.html')

    # Save the first few spatial modes.
    mode_count = 6
    for mode_number in range(mode_count):
        spatial_mode = u[:, mode_number]
        filename = f'spatial_mode_{image_type}_{mode_number}.png'
        _save_image(spatial_mode, filename, image_type)

    # Plot the coefficients for the first few spatial modes for all images.
    x_projected = u.T.dot(x)
    mode_count = 2
    num_images = vh.shape[1]
    images = np.reshape(np.asarray(range(num_images)), (1, num_images))
    legend = [f'Mode {i + 1}' for i in range(mode_count)]
    utils_graph.graph_overlapping_lines(
        np.repeat(images, mode_count, axis=0),
        x_projected,
        legend,
        'Images', 'Coefficient of mode', 
        f'Coefficients of spatial modes for all {image_type} images', 
        plots_dir, 
        f'coef_{image_type}.html')

    print(f'Done analyzing SVD of {image_type} images.')


def reduce_images(x: np.ndarray, plots_dir: str, image_type: str) -> None:
    """Reduces images to just a few spatial modes. 
    """
    print(f'Reducing {image_type} images.')
    (u, s, vh) =  np.linalg.svd(x, full_matrices=False)
    print(f'Shape of U: {u.shape}')
    print(f'Shape of Sigma: {s.shape}')
    print(f'Shape of V*: {vh.shape}')

    # We'll pick a photo to analyze and save the original.
    image_index = 0
    image = x[:, image_index]
    filename = f'reduced_{image_type}_original.png'
    _save_image(image, filename, image_type)

    # We'll analyze the recreated image when we retain just a few modes.
    num_modes = 50
    for i in range(1, num_modes, 5):
        u_reduced = u[:, :i]
        x_reduced = u_reduced.T.dot(x)
        image_reduced = x_reduced[:, image_index]
        image = u_reduced.dot(image_reduced)
        filename = f'reduced_{image_type}_{i}_modes.png'
        _save_image(image, filename, image_type)

    print(f'Done reducing {image_type} images')


def main() -> None:
    """Main program.
    """
    plots_dir = utils_io.find_or_create_dir(PLOTS_FOLDER)
    preprocess_data()
    (xc, xc_labels, xu, xu_labels_1, xu_labels_2) = read_data()
    svd_analysis(xc, plots_dir, image_type='cropped')
    svd_analysis(xu, plots_dir, image_type='uncropped')
    reduce_images(xc, plots_dir, image_type='cropped')
    reduce_images(xu, plots_dir, image_type='uncropped')


if __name__ == '__main__':
    main()
