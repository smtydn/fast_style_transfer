import os
import base64
import io
import time

from flask import Flask, render_template, send_file, request
import PIL

from src import settings
from src.api import utils
from src.transformer.style_transformer import StyleTransformer


def create_app(test_config=None):
    app = Flask(__name__)
    app.template_folder = settings.TEMPLATES_DIR
    app.static_folder = settings.STATICFILE_DIR

    transformer_starry_night = StyleTransformer('starry_night')
    transformer_rain_princess = StyleTransformer('rain_princess')
    transformer_la_muse = StyleTransformer('la_muse')

    @app.route('/', methods=['GET'])
    def index_page():
        return render_template('index.html')

    @app.route('/starry_night', methods=['GET', 'POST'])
    def starry_night():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            filename, save_path = utils.create_temp_img_path()
            utils.remove_temp_images()
            transformer_starry_night.predict(image=uploaded_image, return_decoded=False, save_path=save_path)
            return render_template('generated.html', image=filename)
        return render_template('generated.html')

    @app.route('/rain_princess', methods=['GET', 'POST'])
    def rain_princess():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            filename, save_path = utils.create_temp_img_path()
            utils.remove_temp_images()
            transformer_rain_princess.predict(image=uploaded_image, return_decoded=False, save_path=save_path)
            return render_template('generated.html', image=filename)
        return render_template('generated.html')

    @app.route('/la_muse', methods=['GET', 'POST'])
    def la_muse():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            filename, save_path = utils.create_temp_img_path()
            utils.remove_temp_images()
            transformer_la_muse.predict(image=uploaded_image, return_decoded=False, save_path=save_path)
            return render_template('generated.html', image=filename)
        return render_template('generated.html')

    return app