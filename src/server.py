from flask import Flask, render_template

import settings


def create_app(test_config=None):
    app = Flask(__name__)
    app.template_folder = settings.TEMPLATES_DIR
    app.static_folder = settings.STATICFILE_DIR

    @app.route('/')
    def index_page():
        return render_template('index.html', template_folder=settings.TEMPLATES_DIR)

    @app.route('/test')
    def test_page():
        return render_template('index.html', template_folder=settings.TEMPLATES_DIR)    

    return app