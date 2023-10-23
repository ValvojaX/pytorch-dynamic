import logging
from typing import Any

import torch


class Network(torch.nn.Module):
    def __init__(self, network: torch.nn.Sequential) -> None:
        super(Network, self).__init__()
        self.network = network

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Model created with {len(self.network)} layers. {self}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"Forward pass input {input} ({input.shape}) data type {input.dtype}")
        output = input
        for layer in self.network:
            output = layer(output)

            if isinstance(layer, torch.nn.LSTM):
                output = output[0]

            if hasattr(output, "shape") and hasattr(output, "dtype"):
                self.logger.debug(f"Forward pass output {output} ({output.shape}) data type {output.dtype}")

        return output

    def __str__(self) -> str:
        string = ""
        for layer in self.network:
            if hasattr(layer, "in_features"):
                string += f"{layer.in_features} : "
            if hasattr(layer, "out_features"):
                string += f"{layer.out_features} : "
        return string[:-3]


class NetworkBuilder:
    def __init__(self) -> None:
        self.layers: list[torch.nn.Module] = []

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"NetworkBuilder created.")

    @staticmethod
    def create() -> "NetworkBuilder":
        return NetworkBuilder()

    def add_linear_layer(self, in_features: int, out_features: int, bias: bool = True, *args: Any, **kwargs: Any) -> "NetworkBuilder":
        self.layers.append(torch.nn.Linear(in_features, out_features, bias=bias, *args, **kwargs))
        return self

    def add_lstm_layer(self, in_features: int, out_features: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, *args: Any, **kwargs: Any) -> "NetworkBuilder":
        self.layers.append(torch.nn.LSTM(in_features, out_features, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, *args, **kwargs))
        return self

    def add_dropout_layer(self, p: float = 0.5, inplace: bool = False) -> "NetworkBuilder":
        self.layers.append(torch.nn.Dropout(p=p, inplace=inplace))
        return self

    def add_relu_layer(self) -> "NetworkBuilder":
        self.layers.append(torch.nn.ReLU())
        return self

    def add_sigmoid_layer(self) -> "NetworkBuilder":
        self.layers.append(torch.nn.Sigmoid())
        return self

    def add_tanh_layer(self) -> "NetworkBuilder":
        self.layers.append(torch.nn.Tanh())
        return self

    def add_softmax_layer(self) -> "NetworkBuilder":
        self.layers.append(torch.nn.Softmax())
        return self

    def build(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(*self.layers)

    def build_model(self) -> "Network":
        return Network(self.build())
