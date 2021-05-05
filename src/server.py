import os
import base64
import io

from flask import Flask, render_template, send_file, request
import PIL

import settings
from src.style_transformer import StyleTransformer


def create_app(test_config=None):
    app = Flask(__name__)
    app.template_folder = settings.TEMPLATES_DIR
    app.static_folder = settings.STATICFILE_DIR

    transformer = StyleTransformer('starry_night')
    transformer2 = StyleTransformer('rain_princess')

    @app.after_request
    def add_header(response):
        response.headers['Cache-Control'] = 'public, max-age=0'
        return response

    @app.route('/', methods=['GET', 'POST'])
    def index_page():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            if uploaded_image.content_type != 'application/octet-stream':
                image = transformer2.predict(image=uploaded_image)
                return render_template('generated.html', image=image)
        return render_template('index.html')

    @app.route('/test')
    def test_page():
        image = transformer.predict(r'C:\Users\samet\Projects\fast_style_transfer\images\content\a_man_on_the_moon.jpg')
        return render_template('generated.html', image=image)

    @app.route('/test2', methods=['GET', 'POST'])
    def test_page2():
        if request.method == 'POST':
            uploaded_image = request.files['uploader']
            if uploaded_image.content_type != 'application/octet-stream':
                image = transformer2.predict(image=uploaded_image)
                return render_template('generated.html', image=image)
        return render_template('generated.html')

    return app