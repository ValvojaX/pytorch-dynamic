## Pytorch Dynamic


### 1. Usage

```py
import torch
import logging


from datamodel.definition import DataModelDefinitionBuilder
from datamodel.feature import DataModelFeature
from dataset.dataset import Dataset
from network.network import NetworkBuilder
from scaler.scaler import Scaler
from trainer.trainer import TrainerBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)

# Define a data model
class Coordinate2D:
    # Constructor must have default values for all parameters.
    def __init__(self, x: float = None, y: float = None) -> None:
        self.x: float | None = x
        self.y: float | None = y

    
def main():
    # Create a scaler for data. Different scalers can be used for different features.
    scaler = Scaler.get_instance(name="default")
    
    # Create a data model definition with input and target features.
    definition = DataModelDefinitionBuilder.create() \
        .set_constructor(Coordinate2D) \
        .add_input_feature(DataModelFeature(name="x", scaler=scaler)) \
        .add_target_feature(DataModelFeature(name="y", scaler=scaler)) \
        .build()

    # Create a dataset with data and the data model definition.
    dataset = Dataset(
        data=[Coordinate2D(x=x, y=-x) for x in range(100)],
        model_definition=definition,
        dtype=torch.float32)
    
    # Create a network matching the data model definition.
    model = NetworkBuilder.create() \
        .add_linear_layer(in_features=1, out_features=1, bias=True) \
        .build_model()
    
    # Create a trainer for the network and the dataset.
    trainer = TrainerBuilder.create() \
        .set_model(model=model) \
        .set_optimizer_adam(lr=0.01, weight_decay=0.0001) \
        .set_criterion_mse() \
        .set_split_dataset(dataset=dataset, train_percentage=0.8) \
        .set_model_definition(model_definition=definition) \
        .set_epochs(100) \
        .set_batch_size(2) \
        .build()

    # Train the network.
    trainer.train()
    
    # Save the network.
    trainer.save(path="model.pt")
    
    # Load the network.
    trainer.load(path="model.pt")
    
    # Predict the target feature for a given input feature.
    predictions = [trainer.predict(Coordinate2D(x=x)) for x in range(100, 200)]
    
    # Plot the predictions. Plotting might not work on all data models.
    trainer.plot(predictions=predictions)


if __name__ == "__main__":
    main()
```