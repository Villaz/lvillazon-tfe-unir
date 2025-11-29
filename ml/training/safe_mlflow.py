import os
import tempfile

import mlflow
import mlflow.exceptions
import logging

import numpy as np
from mlflow.models import infer_signature, ModelSignature

logger = logging.getLogger(__name__)


class SafeMLflow:

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.enabled = True
        os.environ["MLFLOW_ARTIFACT_ROOT"] = tempfile.mkdtemp()
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment("birds")
            mlflow.keras.autolog()
        except Exception as e:
            logger.warning(f"MLflow not accesible ({e})")
            self.enabled = False

    def start_run(self, *args, **kwargs):
        if not self.enabled:
            return DummyRun()
        try:
            return mlflow.start_run(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failing in start_run: {e}.")
            self.enabled = False
            return DummyRun()

    def infer_signature(self, model, img_size: tuple[int, int]) -> ModelSignature | None:
        example_input = np.random.rand(1, img_size[0], img_size[1], 3).astype(np.float32)
        example_output = model.predict(example_input)
        if self.enabled:
            return infer_signature(example_input, example_output)
        else:
            return None

    def log_metric(self, *args, **kwargs):
        if not self.enabled:
            return
        try:
            mlflow.log_metric(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failing in log_metric: {e}.")
            self.enabled = False

    def log_param(self, *args, **kwargs):
        if not self.enabled:
            return
        try:
            mlflow.log_param(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failing in log_param: {e}.")
            self.enabled = False


class DummyRun:
    """
    Returns an empty context when the connection is not available
    """
    """Devuelve un contexto vacío para `with mlflow.start_run():`."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
