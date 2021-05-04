import os

# Directories
ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
CHECKPOINTS_DIR = os.path.join(WEIGHTS_DIR, 'checkpoints')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
CONTENT_IMAGES_DIR = os.path.join(IMAGES_DIR, 'content')
STYLE_IMAGES_DIR = os.path.join(IMAGES_DIR, 'style')
OUTPUT_IMAGES_DIR = os.path.join(IMAGES_DIR, 'output')
SAMPLE_IMAGES_DIR = os.path.join(IMAGES_DIR, 'samples')
TEMPLATES_DIR = os.path.join(ROOT_DIR, 'templates')
STATICFILE_DIR = os.path.join(TEMPLATES_DIR, 'static')

# Weights
VGG16_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'vgg16.h5')

# Dataset path
TRAIN_DATASET_PATH = r'D:\Datasets\COCO2014'