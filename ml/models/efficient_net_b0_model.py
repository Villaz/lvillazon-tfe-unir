from abc import ABC
from typing import Any
from keras.src.applications.efficientnet import EfficientNetB0, preprocess_input

from db.bird_dataset import BirdDataset
from models.bird_base_model import BirdModel
import tensorflow as tf
from tensorflow.keras import layers
import loguru

logger = loguru.logger


class BirdEfficientNetB0Model(BirdModel, ABC):

    def __init__(self, dataset: BirdDataset,
                 name: str
                 ):
        super().__init__(dataset, name)



    def load_model(self, weights: str = "imagenet"):
        self.model = EfficientNetB0(
            input_shape=self.dataset.img_size + (3,),
            include_top=False,
            weights=None
        )
        #Debido a un bug en la versión actual de keras, es necesario cargar los pesos manualmente
        logger.info(f"Loading weights from URL")
        weights_url = "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
        weights_path = tf.keras.utils.get_file("efficientnetb0_notop.h5", origin=weights_url, cache_subdir="models")
        self.model.load_weights(weights_path)

    def _preprocess(self, x:Any):
        return preprocess_input(x)

    def _compile(self) -> None:
        self.model = tf.keras.Sequential([self.model,
                                     layers.GlobalAveragePooling2D(),
                                     layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                                     layers.BatchNormalization(),
                                     layers.Activation('relu'),
                                     layers.Dropout(0.5),
                                     layers.Dense(200, activation="softmax")])

        for layer in self.model.layers[0].layers:
            if hasattr(layer, "kernel_regularizer"):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)