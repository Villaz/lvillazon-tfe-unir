from abc import ABC
from typing import Any
from keras.src.applications.resnet import ResNet50, preprocess_input as resnet_preprocess

from db.bird_dataset import BirdDataset
from models.bird_base_model import BirdModel
import tensorflow as tf
from tensorflow.keras import layers


class BirdResNet50Model(BirdModel, ABC):

    def __init__(self,
                 dataset: BirdDataset,
                 name: str
                 ):
        super().__init__(dataset, name)



    def load_model(self, weights: str = "imagenet"):
        self.model = ResNet50(
            input_shape=self.dataset.img_size + (3,),
            include_top=False,
            weights=weights
        )

    def _preprocess(self, x:Any):
        return resnet_preprocess(x)

    def _compile(self) -> None:
        self.model = tf.keras.Sequential([self.model,
                                          layers.GlobalAveragePooling2D(),
                                          layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                                          layers.BatchNormalization(),
                                          layers.Activation('relu'),
                                          layers.Dropout(0.5),
                                          layers.Dense(200, activation="softmax")])