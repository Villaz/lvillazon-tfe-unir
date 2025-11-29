from enum import Enum
import loguru
from .bird_base_model import BirdModel
from .mobilev2_model import BirdMobileNetV2Model
from .efficient_net_b0_model import BirdEfficientNetB0Model
from .resnet50_model import BirdResNet50Model

loguru.logger.enable('tensorflow')
logger = loguru.logger

class ModelType(Enum):
    MOBILENETV2 = BirdMobileNetV2Model
    EFFICIENTNETB0 = BirdEfficientNetB0Model
    RESTNET50 = BirdResNet50Model


def get_model(model_type: ModelType) -> type[BirdModel]:
    return model_type.value

def get_model_from_str(model_name: str) -> ModelType:
    model = None
    if model_name.upper() == "EFFICIENTNETB0":
        model = ModelType.EFFICIENTNETB0
    elif model_name.upper() == "RESTNET50":
        model = ModelType.RESTNET50
    else:
        model = ModelType.MOBILENETV2
    logger.info(f"Using model {model.name}")
    return model