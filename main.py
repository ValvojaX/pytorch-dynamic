# Imports
import json
import math

import torch
import logging


from datamodel.definition import DataModelDefinitionBuilder
from datamodel.feature import DataModelFeature
from dataset.dataset import Dataset
from network.network import NetworkBuilder
from scaler.scaler import Scaler
from trainer.trainer import TrainerBuilder

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)


class Coordinate2D:
    def __init__(self, x: float = None, y: float = None) -> None:
        self.x: float | None = x
        self.y: float | None = y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Coordinate2D({self.x}, {self.y})"


# Main
def main():
    scaler = Scaler.get_instance()
    definition = DataModelDefinitionBuilder.create() \
        .set_constructor(Coordinate2D) \
        .add_input_feature(DataModelFeature(name="x", scaler=scaler)) \
        .add_target_feature(DataModelFeature(name="y", scaler=scaler)) \
        .build()

    dataset = Dataset(
        data=[Coordinate2D(x=x*0.1, y=math.sin(0.1*x)) for x in range(100)],
        model_definition=definition,
        dtype=torch.float32)
    model = NetworkBuilder.create() \
        .add_linear_layer(in_features=1, out_features=1024, bias=True) \
        .add_relu_layer() \
        .add_linear_layer(in_features=1024, out_features=1024, bias=True) \
        .add_relu_layer() \
        .add_linear_layer(in_features=1024, out_features=1, bias=True) \
        .build_model()
    trainer = TrainerBuilder.create() \
        .set_model(model=model) \
        .set_model_definition(model_definition=definition) \
        .set_optimizer_adam(lr=1e-5, weight_decay=0.0001) \
        .set_criterion_mse() \
        .set_split_dataset(dataset=dataset, train_percentage=0.8) \
        .set_epochs(500) \
        .set_batch_size(1) \
        .build()

    trainer.train()
    trainer.save(path="model.pt")
    # trainer.load(path="model.pt")
    predictions = [trainer.predict(Coordinate2D(x=x)) for x in range(100)]
    trainer.plot(predictions=predictions)


if __name__ == "__main__":
    main()
