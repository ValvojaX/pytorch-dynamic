from typing import Any

from scaler.scaler import Scaler


class DataModelFeature:
    def __init__(self, name: str, scaler: Scaler):
        self.name: str = name
        self.scaler: Scaler = scaler

    @property
    def getter(self) -> callable:
        return lambda obj: getattr(obj, self.name)

    @property
    def setter(self) -> callable:
        return lambda obj, value: setattr(obj, self.name, value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scaler": self.scaler.state_dict()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.name = state_dict["name"]
        self.scaler.load_state_dict(state_dict["scaler"])
