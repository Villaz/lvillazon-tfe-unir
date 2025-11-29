import click
import loguru
import tensorflow as tf
from db.bird_dataset import BirdDataset
from models import ModelType, get_model, get_model_from_str
from safe_mlflow import SafeMLflow

logger = loguru.logger


def run(model_name: str,
        model_type: ModelType,
        images_path: str,
        epochs: int):
    gpus = tf.config.list_physical_devices('GPU')
    print("Detected GPUs:", gpus)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    dataset = BirdDataset(images_path, img_size=(224, 224))
    dataset.load(seed=42,
                 batch_size=32,
                 val_split=0.2)

    ml = SafeMLflow("http://localhost:5010")
    with ml.start_run(run_name=model_name):
        model = get_model(model_type)(dataset,
                                 name=model_name)
        model.compile()
        history = model.train(epochs=epochs)
        model.save()

@click.command()
@click.argument("model_name", type=click.STRING)
@click.argument("images_path", type=click.Path(exists=True))
@click.argument("model_type", type=click.STRING)
@click.option("--epochs", type=click.INT, default=30)
def execution(model_name: str,
              images_path: str,
              model_type: str,
              epochs: int):
    logger.info(f"{model_name}:{images_path}:{model_type}:{epochs}")
    run(model_name,
        get_model_from_str(model_type),
        images_path,
        epochs)

if __name__ == "__main__":
    execution()