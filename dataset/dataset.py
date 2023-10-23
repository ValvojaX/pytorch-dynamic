import logging

import numpy as np
import torch

from datamodel.definition import DataModelDefinition


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: list[object], model_definition: DataModelDefinition, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), dtype: torch.dtype = torch.float32):
        self.data: list[object] = data
        self.model_definition: DataModelDefinition = model_definition
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Dataset created with {len(self.data)} data points.")

        self.__initialize_scalers()

    def __initialize_scalers(self) -> None:
        for feature in self.model_definition.input_features:
            feature.scaler.fit([feature.getter(data_point) for data_point in self.data])

        for feature in self.model_definition.target_features:
            feature.scaler.fit([feature.getter(data_point) for data_point in self.data])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        data_point = self.data[index]
        input_data, target_data = np.array([]), np.array([])

        for feature in self.model_definition.input_features:
            input_data = np.append(input_data, feature.scaler.transform([[feature.getter(data_point)]]))

        for feature in self.model_definition.target_features:
            target_data = np.append(target_data, feature.scaler.transform([[feature.getter(data_point)]]))

        return torch.tensor(input_data, dtype=self.dtype).to(self.device), torch.tensor(target_data, dtype=self.dtype).to(self.device)
