import os
import base64
import io
import time

from flask import Flask, render_template, send_file, request
import PIL

from src import settings
from src.api import utils
from src.transformer import StyleTransformer


def create_app(test_config=None):
    app = Flask(__name__)
    app.template_folder = settings.TEMPLATES_DIR
    app.static_folder = settings.STATICFILE_DIR

    # Prediction models
    starry_night = StyleTransformer(r'weights\starry_night.h5')
    rain_princess = StyleTransformer(r'weights\rain_princess.h5')
    la_muse = StyleTransformer(r'weights\la_muse-sw_1-cw_1.h5')

    @app.route('/', methods=['GET'])
    def index_page():
        return render_template('index.html')

    @app.route('/starry_night', methods=['GET', 'POST'])
    def starry_night_route():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            filename, save_path = utils.create_temp_img_path()
            utils.remove_temp_images()
            starry_night.predict(image=uploaded_image, save_path=save_path)
            return render_template('generated.html', image=filename)
        return render_template('generated.html')

    @app.route('/rain_princess', methods=['GET', 'POST'])
    def rain_princess_route():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            filename, save_path = utils.create_temp_img_path()
            utils.remove_temp_images()
            rain_princess.predict(image=uploaded_image, save_path=save_path)
            return render_template('generated.html', image=filename)
        return render_template('generated.html')

    @app.route('/la_muse', methods=['GET', 'POST'])
    def la_muse_route():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            filename, save_path = utils.create_temp_img_path()
            utils.remove_temp_images()
            la_muse.predict(image=uploaded_image, save_path=save_path)
            return render_template('generated.html', image=filename)
        return render_template('generated.html')

    return app