from abc import abstractmethod, ABC
import loguru
import tensorflow as tf
from typing import Any
import numpy as np
from keras import metrics
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.metrics import precision_score ,recall_score, f1_score
from tensorflow.data import Dataset
from tensorflow.keras import layers
from tensorflow.keras.callbacks import History, EarlyStopping
from db.bird_dataset import BirdDataset

loguru.logger.enable('tensorflow')
logger = loguru.logger


class BirdModel(ABC):

    model = None
    classes:list[str] = []
    dataset: BirdDataset = None

    def __init__(self,
                 dataset: BirdDataset,
                 name: str
                 ):
        self.name = name
        self.dataset = dataset
        self.load_model()


    @abstractmethod
    def load_model(self, weights:str = "imagenet") -> None:
        """
        Realiza la carga del modelo a utilizar según la arquitectura indicada
        :param weights: Pesos a utilizar para el modelo
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, dataset:Dataset) -> Dataset:
        """
        Aplica el preprocesamiento necesario para la arquitectura utilizada
        :param dataset: Dataset sobre el cual se aplica el preprocesamiento
        :return: Dataset pre-procesado
        """
        raise NotImplementedError


    @abstractmethod
    def _compile(self)-> None:
        raise NotImplementedError

    def compile(self):
        """
        Realiza la compilación del modelo, utilizando transfer-learning
        :return:
        """
        # Se congelan las capas inferiores del modelo para mantener los pesos y no entrenarlas con los nuevos datos
        # de esta forma se reutilizan los pesos y se disminuye el tiempo necesario de entrenamiento
        self.model.trainable = False
        # Se añaden en las capas superiores del modelo las capas específicas del modelo desarrollado
        self._compile()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss= "categorical_crossentropy",
            metrics=["accuracy",
                     metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
                     ]
        )
        self.model.summary()

    def compile_finetune(self):
        """
        Compila el modelo para realizar un finetune
        Para ello se descongelan las últimas 50 capas del modelo
        y se reentrenan.
        :return:
        """
        fine_tune_at = len(self.model.layers) - 50

        for i, layer in enumerate(self.model.layers):
            layer.trainable = i >= fine_tune_at

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=1e-4,
                momentum=0.9
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy",
                     metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
                     ]
        )
        self.model.summary()


    def train(self, epochs: int) -> History:
        """
        Realiza el entrenamiento del modelo
        :param epochs: Número de épocas de entrenamiento
        :return:
        """
        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True,
                                   min_delta=0.0005),
                     ReduceLROnPlateau(
                         monitor='val_loss',
                         factor=0.3,
                         patience=5,
                         min_lr=1e-6
                     )
                     ]

        self.dataset.normalize(self._preprocess)
        history = self.model.fit(
            self.dataset.train_ds,
            validation_data=self.dataset.val_ds,
            epochs=epochs,
            callbacks=[callbacks]
        )

        self.model.save(f"{self.name}.keras")
        return history

    def save(self) -> None:
        """
        Salva el modelo generado
        :return:
        """
        self.model.save(f"{self.name}.keras", zipped=True)

    def predict(self, x) -> Any:
        """
        Realiza predicciones sobre el modelo.
        :param x: Dataset con los datos a predecir
        :return:
        """
        return self.model.predict(x)

    def get_f1_recall_precision(self) -> dict:
        """
        Calcula las métricas de f1, recall y precision sobre
        el dataset de validación
        :return:
        """
        final_labels = []
        final_predictions = []
        for images, labels in self.dataset.val_ds:
            final_labels.extend(np.argmax(labels, axis=1))
            predictions = self.model.predict(images)
            final_predictions.extend(np.argmax(predictions, axis=1))
        return {'recall': recall_score(final_labels, final_predictions, average='weighted'),
        'f1': f1_score(final_labels, final_predictions, average='weighted'),
        'precision' : precision_score(final_labels, final_predictions, average='weighted')
                }

