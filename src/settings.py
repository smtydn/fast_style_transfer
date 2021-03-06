import os
import pathlib

# Directories
ROOT_DIR = pathlib.Path(os.path.split(os.path.abspath(__file__))[0]).parent.absolute()
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
CHECKPOINTS_DIR = os.path.join(WEIGHTS_DIR, 'checkpoints')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
CONTENT_IMAGES_DIR = os.path.join(IMAGES_DIR, 'content')
STYLE_IMAGES_DIR = os.path.join(IMAGES_DIR, 'style')
OUTPUT_IMAGES_DIR = os.path.join(IMAGES_DIR, 'outputs')
SAMPLE_IMAGES_DIR = os.path.join(IMAGES_DIR, 'samples')
TEMPLATES_DIR = os.path.join(ROOT_DIR, 'templates')
STATICFILE_DIR = os.path.join(TEMPLATES_DIR, 'static')
STATICFILE_IMAGES_DIR = os.path.join(STATICFILE_DIR, 'images')

# Weights
VGG16_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'vgg16.h5')

# Dataset path
TRAIN_DATASET_PATH = r'D:\Datasets\COCO2014'

# Available weights for production
AVAILABLE_STYLES = {
    'starry_night': os.path.join(WEIGHTS_DIR, 'starry_night.h5'),
    'rain_princess': os.path.join(WEIGHTS_DIR, 'rain_princess.h5'),
    'la_muse': os.path.join(WEIGHTS_DIR, 'la_muse-sw_1-cw_1.h5')
}
