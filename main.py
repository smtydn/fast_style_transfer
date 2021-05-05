import click

@click.group()
def cli():
    pass


@cli.command()
@click.option('--learning_rate', default=1e-3, show_default=True)
@click.option('--style_image', help='Name of the style image. Should exists under images/style. Name should have extension.')
@click.option('--batch_size', default=4, show_default=True)
@click.option('--image_size', default=(256, 256), show_default=True)
@click.option('--content_weight', default=1e4, show_default=True)
@click.option('--style_weight', default=1e-2, show_default=True)
@click.option('--log_interval', default=100, show_default=True)
@click.option('--chkpt_interval', default=2000, show_default=True)
@click.option('--epochs', default=1, show_default=True)
@click.option('--sample_interval', default=1000, show_default=True)
@click.option('--content_image')
@click.option('--tv_weight', default=1e-6, show_default=True)
@click.option('--chkpt_path', type=click.Path(), required=False, default=None)
def train(learning_rate, style_image, batch_size, image_size, content_weight, style_weight, log_interval, chkpt_interval, epochs,
        sample_interval, content_image, tv_weight, chkpt_path):
    from src.train import start
    start(
        learning_rate, style_image, batch_size, image_size, content_weight, style_weight, log_interval, chkpt_interval, epochs,
        sample_interval, content_image, tv_weight, chkpt_path
    )


@cli.command()
@click.option('--style_name')
@click.option('--content_path', type=click.Path())
@click.option('--output_path', type=click.Path())
@click.option('--weights_path', type=click.Path(), default=None, show_default=True)
def transform(style_name, content_path, output_path, weights_path):
    from src.style_transformer import StyleTransformer
    transformer = StyleTransformer(style_name, weights_path=weights_path)
    transformer.predict(content_path, save_path=output_path, return_decoded=False)


@cli.command()
def runserver():
    from src.server import create_app
    app = create_app()
    app.run()


if __name__ == '__main__':
    cli()