# Imports
import torch
import logging

from typing import Any, List

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Scaler
class Scaler(MinMaxScaler):
    _instance = None

    def __new__(cls, *args, **kwargs) -> "Scaler":
        if Scaler._instance is None:
            Scaler._instance = super(Scaler, cls).__new__(cls, *args, **kwargs)
        return Scaler._instance

    def __init__(self) -> None:
        if not hasattr(self, "initialized"):
            super(Scaler, self).__init__()
            self.initialized = True
            self._min = float("inf")
            self._max = float("-inf")

            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Scaler created with min {self._min} and max {self._max}")

    @staticmethod
    def get_instance() -> "Scaler":
        return Scaler()

    def fit(self, X: List[Any], y=None) -> object:
        self._min = min(self._min, min(X))
        self._max = max(self._max, max(X))
        return super(Scaler, self).fit([[self._min], [self._max]])

    def transform(self, X: Any) -> Any:
        return super(Scaler, self).transform(X)

    def inverse_transform(self, X: Any) -> Any:
        return super(Scaler, self).inverse_transform(X)

    def state_dict(self) -> dict[str, float]:
        return {
            "min": self._min,
            "max": self._max
        }

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        self._min = state_dict["min"]
        self._max = state_dict["max"]


# Data
class CoordinateData:
    scaler = Scaler.get_instance()

    def __init__(self, x_coordinate: int, y_coordinate: int) -> None:
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        CoordinateData.scaler.fit([x_coordinate, y_coordinate])

    def __str__(self) -> str:
        return f"({self.x_coordinate}, {self.y_coordinate})"

    def __eq__(self, other) -> bool:
        if isinstance(other, CoordinateData):
            return self.x_coordinate == other.x_coordinate and self.y_coordinate == other.y_coordinate
        return False

    def __hash__(self) -> int:
        return hash((self.x_coordinate, self.y_coordinate))

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(CoordinateData.scaler.transform([[self.x_coordinate]]), dtype=torch.float32)
        y = torch.tensor(CoordinateData.scaler.transform([[self.y_coordinate]]), dtype=torch.float32)
        return x.to(DEVICE), y.to(DEVICE)

    def from_tensors(self) -> "CoordinateData":
        x = CoordinateData.scaler.inverse_transform([[self.x_coordinate]])
        y = CoordinateData.scaler.inverse_transform([[self.y_coordinate]])
        return CoordinateData(x, y)


# Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: list[CoordinateData]) -> None:
        self.data: list[CoordinateData] = data
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Dataset created with {len(self.data)} data points.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index].to_tensors()

    @staticmethod
    def generate() -> "Dataset":
        data = []
        for i in range(100):
            data.append(CoordinateData(i, i))
        return Dataset(data)


# Network builder
class NetworkBuilder:
    def __init__(self) -> None:
        self.layers: list[torch.nn.Module] = []

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"NetworkBuilder created.")

    @staticmethod
    def create() -> "NetworkBuilder":
        return NetworkBuilder()

    def add_linear_layer(self, in_features: int, out_features: int, bias: bool = True, *args: Any, **kwargs: Any) -> "NetworkBuilder":
        self.layers.append(torch.nn.Linear(in_features, out_features, bias=bias, *args, **kwargs))
        return self

    def add_relu_layer(self) -> "NetworkBuilder":
        self.layers.append(torch.nn.ReLU())
        return self

    def build(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(*self.layers)

    def build_model(self) -> "Model":
        return Model(self.build())


# Model
class Model(torch.nn.Module):
    def __init__(self, network: torch.nn.Sequential) -> None:
        super(Model, self).__init__()
        self.network = network

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Model created with {len(self.network)} layers. {self.network[0].in_features} -> {self.network[-1].out_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        self.logger.debug(f"Forward pass input {x} ({x.shape}) output {output} ({output.shape})")
        return output


# Trainer builder
class TrainerBuilder:
    def __init__(self):
        self.trainer = Trainer()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TrainerBuilder created.")

    @staticmethod
    def create() -> "TrainerBuilder":
        return TrainerBuilder()

    def set_model(self, model: torch.nn.Module) -> "TrainerBuilder":
        self.trainer.model = model.to(DEVICE)
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

    def set_scaler(self, scaler: Scaler) -> "TrainerBuilder":
        self.trainer.scaler = scaler
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

    def build(self) -> "Trainer":
        return self.trainer


# Trainer
class Trainer:
    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.criterion: torch.nn.Module = torch.nn.MSELoss()
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler: Scaler | None = None

        self.batch_size: int = 1
        self.epochs: int = 100
        self.max_patience: int = 0

        self.train_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Trainer created with {DEVICE} device.")

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

    def predict(self, x: int) -> CoordinateData:
        # Check if model is ready
        self.__check_ready()

        # Set model to eval mode
        self.model.eval()

        # Predict
        with torch.no_grad():
            tensor = torch.tensor(CoordinateData.scaler.transform([[x]]), dtype=torch.float32).to(DEVICE)
            output = self.model(tensor)
            return CoordinateData(x, CoordinateData.scaler.inverse_transform(output.cpu().numpy())[0][0])

    def save(self, path: str) -> None:
        # Check if model is ready
        self.__check_ready()

        # save model, optimizer, and scaler
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict()
        }, path)

    def load(self, path: str) -> None:
        save = torch.load(path)
        self.model.load_state_dict(save["model_state_dict"])
        self.optimizer.load_state_dict(save["optimizer_state_dict"])
        CoordinateData.scaler.load_state_dict(save["scaler"])

    def __check_ready(self) -> None:
        error_msg = ""
        match None:
            case self.model:
                error_msg = "No model loaded."
            case self.optimizer:
                error_msg = "No optimizer loaded."
            case self.scaler:
                error_msg = "No scaler loaded."
            case self.train_dataset:
                error_msg = "No train dataset loaded."

        if error_msg:
            self.logger.error(error_msg)
            raise ValueError(error_msg)


# Visualizer
class Visualizer:
    @staticmethod
    def plot(train_data: List[CoordinateData], prediction_data: List[CoordinateData]) -> None:
        plt.plot([data.x_coordinate for data in train_data], [data.y_coordinate for data in train_data], "ro", label="Train Data")
        plt.plot([data.x_coordinate for data in prediction_data], [data.y_coordinate for data in prediction_data], "bo", label="Prediction Data")
        plt.title("Train Data vs Prediction Data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

# weight_decay to optimizers for L2 regularization
# dropout layers for regularization


# Main
def main():
    dataset = Dataset.generate()
    model = NetworkBuilder.create() \
        .add_linear_layer(in_features=1, out_features=1, bias=True) \
        .build_model()
    trainer = TrainerBuilder.create() \
        .set_model(model=model) \
        .set_optimizer_adam(lr=0.01, weight_decay=0.0001) \
        .set_criterion_mse() \
        .set_scaler(scaler=Scaler.get_instance()) \
        .set_split_dataset(dataset=dataset) \
        .set_epochs(100) \
        .set_batch_size(2) \
        .build()

    trainer.train()
    Visualizer.plot(dataset.data, [trainer.predict(i) for i in range(100, 200)])


if __name__ == "__main__":
    main()
