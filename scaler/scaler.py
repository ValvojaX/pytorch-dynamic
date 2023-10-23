import logging
from typing import List, Any

from sklearn.preprocessing import MinMaxScaler


class Scaler(MinMaxScaler):
    _instances: dict[str, "Scaler"] = {}

    def __new__(cls, name: str = "default", *args, **kwargs) -> "Scaler":
        if name not in cls._instances:
            cls._instances[name] = super(Scaler, cls).__new__(cls, *args, **kwargs)
        return cls._instances[name]

    def __init__(self, name: str = "default") -> None:
        if not hasattr(self, "initialized"):
            super(Scaler, self).__init__()
            self.initialized = True
            self.name = name
            self._min = float("inf")
            self._max = float("-inf")

            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"Scaler {name} created with min {self._min} and max {self._max}")

    @staticmethod
    def get_instance(name: str = "default") -> "Scaler":
        return Scaler(name=name)

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
        self.fit([self._min, self._max])
