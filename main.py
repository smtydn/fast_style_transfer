import click

@click.group()
def cli():
    pass


@cli.command()
@click.option('--style_image_path', type=click.Path(), help='Path to the style image.')
@click.option('--content_image_path', type=click.Path(), help='Path to the content image. Will be used for sampling.')
@click.option('--train_dataset_path', type=click.Path(), help='Path to the training dataset.')
@click.option('--checkpoint_dir', type=click.Path(), help='Path for checkpoints.')
@click.option('--weights_dir', type=click.Path(), help='Path for weights to save.')
@click.option('--sample_dir', type=click.Path(), help='Path for sample images to save.')
@click.option('--batch_size', type=click.INT, help="Batch size.", default=4, show_default=True)
@click.option('--epochs', type=click.INT, help='Number of epochs.', default=1, show_default=True)
@click.option('--learning_rate', type=click.FLOAT, help='Learning rate.', default=1e-3, show_default=True)
@click.option('--content_weight', type=click.FLOAT, help='Content loss weight.', default=1.0, show_default=True)
@click.option('--style_weight', type=click.FLOAT, help='Style loss weight.', default=1.0, show_default=True)
@click.option('--tv_weight', type=click.FLOAT, help='Total variation loss weight.', default=1e-6, show_default=True)
@click.option('--log_interval', type=click.INT, help='Logging interval.', default=100, show_default=True)
@click.option('--checkpoint_interval', type=click.INT, help='Checkpoint interval.', default=2000, show_default=True)
@click.option('--sample_interval', type=click.INT, help='Sampling interval.', default=1000, show_default=True)
@click.option('--checkpoint_path', type=click.Path(), required=False, default=None, help='A path for model to load weights before training.')
def train(**kwargs):
    from src.transformer import TransformNetTrainer
    trainer = TransformNetTrainer(**kwargs)
    trainer.run()


@cli.command()
@click.option('--weights_path', type=click.Path(), help='Path for model to load weights.')
@click.option('--save_path', type=click.Path(), help='Path for model to save output from prediction.')
@click.option('--image_path', type=click.Path(), help='Path for content image.')
def predict(weights_path, save_path, image_path):
    from src.transformer import StyleTransformer
    transformer = StyleTransformer(weights_path)
    transformer.predict(save_path, image_path=image_path)


@cli.command()
def runserver():
    from src.api.server import create_app
    app = create_app()
    app.run()


if __name__ == '__main__':
    cli()