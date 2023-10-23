import logging
from typing import Any

import numpy as np
import torch

from datamodel.feature import DataModelFeature
from scaler.scaler import Scaler


class DataModelDefinition:
    def __init__(self, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.input_features: list[DataModelFeature] = []
        self.target_features: list[DataModelFeature] = []
        self.constructor: callable = None
        self.device: torch.device = device

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"DataModelDefinition created.")

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_features": [feature.state_dict() for feature in self.input_features],
            "target_features": [feature.state_dict() for feature in self.target_features]
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.input_features = [DataModelFeature("", Scaler()) for _ in range(len(state_dict["input_features"]))]
        self.target_features = [DataModelFeature("", Scaler()) for _ in range(len(state_dict["target_features"]))]
        for i, feature in enumerate(state_dict["input_features"]):
            self.input_features[i].load_state_dict(feature)
        for i, feature in enumerate(state_dict["target_features"]):
            self.target_features[i].load_state_dict(feature)

    def from_tensors(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> object:
        obj = self.constructor()
        self.apply_input_tensor(obj, input_tensor)
        self.apply_target_tensor(obj, target_tensor)
        return obj

    def get_input_features(self, obj: object) -> list[Any]:
        features = []
        for feature in self.input_features:
            features.append(feature.getter(obj))
        return features

    def get_target_features(self, obj: object) -> list[Any]:
        features = []
        for feature in self.target_features:
            features.append(feature.getter(obj))
        return features

    def get_input_tensor(self, obj: object) -> torch.Tensor:
        features = np.array([])
        for feature in self.input_features:
            features = np.append(features, feature.getter(obj))
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def get_target_tensor(self, obj: object) -> torch.Tensor:
        features = np.array([])
        for feature in self.target_features:
            features = np.append(features, feature.getter(obj))
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def get_input_tensor_scaled(self, obj: object) -> torch.Tensor:
        features = np.array([])
        for feature in self.input_features:
            features = np.append(features, feature.scaler.transform([[feature.getter(obj)]]))
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def get_target_tensor_scaled(self, obj: object) -> torch.Tensor:
        features = np.array([])
        for feature in self.target_features:
            features = np.append(features, feature.scaler.transform([[feature.getter(obj)]]))
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def apply_input_tensor(self, obj: object, input_tensor: torch.Tensor) -> object:
        input_tensor = input_tensor.cpu().detach().numpy()
        for i, feature in enumerate(self.input_features):
            feature.setter(obj, feature.scaler.inverse_transform([[input_tensor[0][i]]])[0][0])
        return obj

    def apply_target_tensor(self, obj: object, target_tensor: torch.Tensor) -> object:
        target_tensor = target_tensor.cpu().detach().numpy()
        for i, feature in enumerate(self.target_features):
            feature.setter(obj, feature.scaler.inverse_transform([[target_tensor[0][i]]])[0][0])
        return obj


class DataModelDefinitionBuilder:
    def __init__(self, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.definition: DataModelDefinition = DataModelDefinition(device=device)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"DataModelDefinitionBuilder created.")

    @staticmethod
    def create(device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> "DataModelDefinitionBuilder":
        return DataModelDefinitionBuilder(device=device)

    def add_input_feature(self, feature: DataModelFeature) -> "DataModelDefinitionBuilder":
        self.definition.input_features.append(feature)
        return self

    def add_target_feature(self, feature: DataModelFeature) -> "DataModelDefinitionBuilder":
        self.definition.target_features.append(feature)
        return self

    def set_constructor(self, constructor: object) -> "DataModelDefinitionBuilder":
        self.definition.constructor = constructor
        return self

    def build(self) -> "DataModelDefinition":
        if self.definition.constructor is None:
            self.logger.warning("Constructor not set.")

        return self.definition
