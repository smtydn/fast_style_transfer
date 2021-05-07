import os
import time

from src import settings


def remove_temp_images():
    for filename in os.listdir(settings.STATICFILE_IMAGES_DIR):
        if filename.startswith('temp_'):
            os.remove(os.path.join(settings.STATICFILE_IMAGES_DIR, filename))


def create_temp_img_path():
    filename = f'temp_{int(time.time())}.png'
    save_path = os.path.join(settings.STATICFILE_IMAGES_DIR, filename)
    return ['images/' + filename, save_path]