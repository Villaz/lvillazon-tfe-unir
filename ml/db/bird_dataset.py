from keras import Sequential, layers
from keras.src.utils import image_dataset_from_directory
from tensorflow import data as tf_data

class BirdDataset:

    classes: list[str]

    def __init__(self, path: str, img_size: tuple):
        self.path = path
        self.img_size = img_size


    def _augmentation_layers(self):
        """
        Aplica capas de distorsión a la entrada del modelo con el fin de aplicar Data Augmentation y
        aumentar la variabilidad de los datos de entrada
        :return:
        """
        augmentation_layers = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1, fill_mode="nearest"),
            layers.RandomZoom(0.1, fill_mode="nearest")
        ], name="augmentation_layers")
        self.train_ds = self.train_ds.map(lambda x, y: (augmentation_layers(x), y))
        self.val_ds = self.val_ds.map(lambda x, y: (augmentation_layers(x), y))

    def load(self,
             seed: int = 42,
             batch_size: int = 32,
             augmentation: bool = False,
             val_split: float = 0.2):
        """
        Realiza la lectura del directorio con los datos de entrada
        :param seed: Seed a utilizar para cargar de manera aleatoria los datos en los datasets de entrenamiento y validación
        :param batch_size: Tamaño del batch a usar en el procesamiento
        :param augmentation: Indica si en las imágenes se va a aplicar data augmentation
        :param val_split: Porcentaje a utilizar para el conjunto de validación
        :return:
        """
        self.train_ds, self.val_ds = image_dataset_from_directory(self.path,
                                                        label_mode='categorical',
                                                        validation_split=val_split,
                                                        subset="both",
                                                        seed=seed,
                                                        image_size=self.img_size,
                                                        batch_size=batch_size)
        self.classes = self.train_ds.class_names
        if augmentation:
            self._augmentation_layers()

    def normalize(self, preprocess):
        """
                Normaliza la entrada aplicando el preprocesamiento necesario para la arquitectura utilizada.
                Realiza un cacheado y un prefetch del dataset para mejorar el rendimiento.
                :return: None
                """
        self.train_ds = self.train_ds.map(lambda x, y: (preprocess(x), y))
        self.val_ds = self.val_ds.map(lambda x, y: (preprocess(x), y))
        # Mejora del rendimiento
        self.train_ds = self.train_ds.prefetch(tf_data.AUTOTUNE).cache()
        self.validation_ds = self.val_ds.prefetch(tf_data.AUTOTUNE).cache()