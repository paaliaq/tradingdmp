"""Base classes for models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseFeatureModel(ABC):
    """Base class for feature based modeling.

    The main difference between BaseFeatureModel and BaseTimeModel is that
    BaseFeatureModel uses data from BaseFeatureData, whereas BaseTimeModel uses data
    from BaseTimeData.
    """

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
        """Method for fitting a model.

        This function should fit the model given training data x and y. This data
        should be feature data, i.e. it should come from BaseFeatureData.

        Args:
            x: Training features in data frame of shape (n, m), where n is the number of
                samples and m is the number of features.
            y: Training targets in data frame of shape (n, d), where n is the number of
                samples and d is the number of target variables.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Method for predicting with a fitted model.

        This function should predict with a model given test data x. This data
        should be feature data, i.e. it should come from BaseFeatureData.

        Args:
            x: Test features in data frame of shape (n, m), where n is the number of
                samples and m is the number of features.

        Returns:
            y: Predicted targets in a numpy array of shape (n, d), where n is the number
                of samples and d is the number of target variables.
        """
        pass


class BaseTimeModel(ABC):
    """Base class for timeseries based modeling.

    The main difference between BaseFeatureModel and BaseTimeModel is that
    BaseFeatureModel uses data from BaseFeatureData, whereas BaseTimeModel uses data
    from BaseTimeData.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> None:
        """Method for fitting a model.

        This function should fit the model given training data x and y. This data
        should be timeseries data, i.e. it should come from BaseTimeData.

        Args:
            x: Training features in numpy array of shape (n, t, m), where n is the
                number of samples, t is the number of timesteps per sample, and m is
                the number of features.
            y: Training targets in numpy array of shape (n, d), where n is the number of
                samples and d is the number of target variables per sample.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Method for predicting with a fitted model.

        This function should predict with a model given test data x. This data
        should be timeseries data, i.e. it should come from BaseTimeData.

        Args:
            x: Test features in numpy array of shape (n, t, m), where n is the
                number of samples, t is the number of timesteps per sample, and m is
                the number of features.

        Returns:
            y: Predicted targets in a numpy array of shape (n, d), where n is the number
                of samples and d is the number of target variables per sample.
        """
        pass
