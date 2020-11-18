"""Base class for all model classes."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """Base class for data fetching and processing."""

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
        """Method for fitting a model.

        This function should fit the model given some training data x and y.

        Args:
            x (pd.DataFrame): Training features in data frame of shape (n, m)
            y (pd.DataFrame): Training targets in data frame of shape (n, d)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Method for predicting with a fitted model.

        This function should fit the model given some training data x and y.

        Args:
            x (pd.DataFrame): Test features in data frame of shape (n, m)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            y: Predicted targets in data frame of shape (n, d)

        """
        pass
