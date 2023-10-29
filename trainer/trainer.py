import logging
from typing import Any

import torch
from matplotlib import pyplot as plt

from datamodel.definition import DataModelDefinition
from dataset.dataset import Dataset

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.criterion: torch.nn.Module = torch.nn.MSELoss()
        self.optimizer: torch.optim.Optimizer | None = None

        self.batch_size: int = 1
        self.epochs: int = 100
        self.max_patience: int = 0

        self.train_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None

        self.model_definition: DataModelDefinition | None = None
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Trainer created with {self.device} device.")

    def train(self) -> None:
        # Check if model is ready
        self.__check_ready()

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        if self.validation_dataset is not None:
            validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True)

        # Early stopping
        patience = 0
        best_loss = float("inf")

        # Train
        for epoch in range(self.epochs):
            train_loss, validation_loss = 0.0, 0.0

            # Train
            self.model.train()
            for inputs, targets in train_loader:
                # forward pass
                outputs = self.model(inputs)

                # calculate loss
                loss = self.criterion(outputs, targets)
                train_loss += loss.item()

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.validation_dataset is None:
                # Log
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}"
                                 f" - Train loss: {train_loss / len(train_loader)}")
                continue

            # Validate
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    # forward pass
                    outputs = self.model(inputs)

                    # calculate loss
                    loss = self.criterion(outputs, targets)
                    validation_loss += loss.item()

            # Log
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}"
                             f" - Train loss: {train_loss / len(train_loader)}"
                             f" - Validation loss: {validation_loss / len(validation_loader)}")

            # Early stopping
            if validation_loss < best_loss:
                best_loss = validation_loss
                patience = 0
            else:
                patience += 1

            if patience >= self.max_patience > 0:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    def predict(self, o: object) -> object:
        # Check if model is ready
        self.__check_ready()

        # Set model to eval mode
        self.model.eval()

        # Predict
        with torch.no_grad():
            definition = self.model_definition
            outputs = self.model(definition.get_input_tensor_scaled(o).unsqueeze(0))
            return definition.apply_target_tensor(o, outputs)

    def save(self, path: str) -> None:
        # Check if model is ready
        self.__check_ready()

        # save model, optimizer, and model definition
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_definition": self.model_definition.state_dict()
        }, path)

    def load(self, path: str) -> None:
        save = torch.load(path)
        self.model.load_state_dict(save["model_state_dict"])
        self.optimizer.load_state_dict(save["optimizer_state_dict"])
        self.model_definition.load_state_dict(save["model_definition"])

    def plot(self, predictions: list[object]) -> None:
        # Check if model is ready
        self.__check_ready()

        definition = self.model_definition
        train_input_features: dict[str, list[float]] = dict()
        train_target_features: dict[str, list[float]] = dict()
        prediction_input_features: dict[str, list[float]] = dict()
        prediction_target_features: dict[str, list[float]] = dict()

        for inputs, targets in self.train_dataset:
            self.logger.debug(f"Plotting train data: {inputs} -> {targets}")
            obj = definition.from_tensors(input_tensor=inputs.unsqueeze(0), target_tensor=targets.unsqueeze(0))
            for feature in definition.input_features:
                if feature.name not in train_input_features:
                    train_input_features[feature.name] = []

                train_input_features[feature.name].append(feature.getter(obj))

            for feature in definition.target_features:
                if feature.name not in train_target_features:
                    train_target_features[feature.name] = []

                train_target_features[feature.name].append(feature.getter(obj))

        for prediction in predictions:
            for feature in definition.input_features:
                if feature.name not in prediction_input_features:
                    prediction_input_features[feature.name] = []

                prediction_input_features[feature.name].append(feature.getter(prediction))

            for feature in definition.target_features:
                if feature.name not in prediction_target_features:
                    prediction_target_features[feature.name] = []

                prediction_target_features[feature.name].append(feature.getter(prediction))

        # Plot
        plt.figure(figsize=(10, 7))

        for input_feature, target_feature in zip(definition.input_features, definition.target_features):
            plt.scatter(train_input_features[input_feature.name], train_target_features[target_feature.name], label="Train Data")
            plt.scatter(prediction_input_features[input_feature.name], prediction_target_features[target_feature.name], label="Prediction Data")

        # Get min and max of all input features
        min_x, max_x = None, None
        for input_feature in definition.input_features:
            for value in train_input_features[input_feature.name]:
                if min_x is None or value < min_x:
                    min_x = value

                if max_x is None or value > max_x:
                    max_x = value

            for value in prediction_input_features[input_feature.name]:
                if min_x is None or value < min_x:
                    min_x = value

                if max_x is None or value > max_x:
                    max_x = value

        plt.xlim(min_x * 1.1, max_x * 1.1)

        # Get min and max of all target features
        min_y, max_y = None, None
        for target_feature in definition.target_features:
            for value in train_target_features[target_feature.name]:
                if min_y is None or value < min_y:
                    min_y = value

                if max_y is None or value > max_y:
                    max_y = value

            for value in prediction_target_features[target_feature.name]:
                if min_y is None or value < min_y:
                    min_y = value

                if max_y is None or value > max_y:
                    max_y = value

        plt.ylim(min_y * 1.1, max_y * 1.1)

        plt.title("Train Data vs Prediction Data")
        plt.xlabel("Input Features")
        plt.ylabel("Target Features")
        plt.legend()
        plt.show()

    def __check_ready(self) -> None:
        error_msg = ""
        match None:
            case self.model:
                error_msg = "No model loaded."
            case self.optimizer:
                error_msg = "No optimizer loaded."
            case self.train_dataset:
                error_msg = "No train dataset loaded."

        if error_msg:
            self.logger.warning(error_msg)


class TrainerBuilder:
    def __init__(self, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.trainer = Trainer()
        self.trainer.device = device

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"TrainerBuilder created.")

    @staticmethod
    def create(device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> "TrainerBuilder":
        return TrainerBuilder(device=device)

    def set_model(self, model: torch.nn.Module) -> "TrainerBuilder":
        self.trainer.model = model.to(self.trainer.device)
        return self

    def set_criterion(self, criterion: torch.nn.Module) -> "TrainerBuilder":
        self.trainer.criterion = criterion
        return self

    def set_criterion_mse(self) -> "TrainerBuilder":
        self.trainer.criterion = torch.nn.MSELoss()
        return self

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> "TrainerBuilder":
        self.trainer.optimizer = optimizer
        return self

    def set_optimizer_adam(self, lr: float = 0.001, weight_decay: float = 0.0, *args: Any, **kwargs: Any) -> "TrainerBuilder":
        self.trainer.optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=lr, weight_decay=weight_decay, *args, **kwargs)
        return self

    def set_optimizer_sgd(self, lr: float = 0.001, momentum: float = 0.0, weight_decay: float = 0.0, *args: Any, **kwargs: Any) -> "TrainerBuilder":
        self.trainer.optimizer = torch.optim.SGD(self.trainer.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, *args, **kwargs)
        return self

    def set_batch_size(self, batch_size: int) -> "TrainerBuilder":
        self.trainer.batch_size = batch_size
        return self

    def set_epochs(self, epochs: int) -> "TrainerBuilder":
        self.trainer.epochs = epochs
        return self

    def set_train_dataset(self, train_dataset: Dataset) -> "TrainerBuilder":
        self.trainer.train_dataset = train_dataset
        return self

    def set_validation_dataset(self, validation_dataset: Dataset) -> "TrainerBuilder":
        self.trainer.validation_dataset = validation_dataset
        return self

    def set_split_dataset(self, dataset: Dataset, train_percentage: float = 0.8) -> "TrainerBuilder":
        train_size = int(train_percentage * len(dataset))
        test_size = len(dataset) - train_size

        self.trainer.train_dataset, self.trainer.validation_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        return self

    def set_model_definition(self, model_definition: DataModelDefinition) -> "TrainerBuilder":
        self.trainer.model_definition = model_definition
        return self

    def build(self) -> "Trainer":
        return self.trainer