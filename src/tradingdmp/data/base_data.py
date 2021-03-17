"""Base classes for data pipelines."""

import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd


class BaseFeatureData(ABC):
    """Base class for feature data fetching and processing.

    The main difference between BaseFeatureData and BaseTimeData is that BaseFeatureData
    gives pandas data frames with feature data, whereas BaseTimeData gives numpy arrays
    with timeseries data.
    """

    @abstractmethod
    def get_data(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        *args: Any,
        **kwargs: Any
    ) -> Union[
        Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], Tuple[pd.DataFrame, pd.DataFrame]
    ]:
        """Method for getting data that can be passed to a model.

        This function should fetch raw data, clean this data, conduct feature
        engineering and split the data into predictors x and targets y. This function
        should return feature data.

        Args:
            ticker_list: A list of ticker symbols for which to get data.
            dt_start: All data after incl. dt_start is fetched.
                By default, dt_start is None, which means that data is fetched
                from the first available datetime onward.
            dt_end: All data until incl. dt_end is fetched.
                By default, dt_end is None, which means that data is fetched
                until the last the available datetime.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            data_dict: There are two options. Option A) must be implemented by default.
                Option B) can optionally be implemented for convenience to get the data
                in a format that is suitable for model fitting.
                Option A): A dictionary that contains a key-value pair for each ticker
                symbol. The key always represents the ticker symbol. There is one key
                for each element in the input ticker_list. Each value is a tuple of
                two pandas data frames x and y: x of shape (n, m) and y of shape (n, d),
                where n is the number of samples, m is the number of features and d is
                the number of target variables.
                Option B): A tuple of two data frames x and y. These data frames contain
                all data points for all dates and tickers.

        """
        pass


class BaseTimeData(ABC):
    """Base class for timeseries data fetching and processing.

    The main difference between BaseFeatureData and BaseTimeData is that BaseFeatureData
    gives pandas data frames with feature data, whereas BaseTimeData gives numpy arrays
    with timeseries data.
    """

    @abstractmethod
    def get_data(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        *args: Any,
        **kwargs: Any
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Method for getting data that can be passed to a model.

        This function should fetch raw data, clean this data, conduct feature
        engineering and split the data into predictors x and targets y. This function
        should return timeseries data.

        Args:
            ticker_list: A list of ticker symbols for which to get data.
            dt_start: All data after incl. dt_start is fetched.
                By default, dt_start is None, which means that data is fetched
                from the first available datetime onward.
            dt_end: All data until incl. dt_end is fetched.
                By default, dt_end is None, which means that data is fetched
                until the last the available datetime.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            data_dict: a dictionary that contains one key-value pair for each
                ticker symbol. The key always represents the ticker symbol. There is one
                key for each element in the input ticker_list. Each value is a tuple of
                two numpy arrays: x of shape (n, t, m) and y of shape (n, d), where n is
                the number of samples, t is the number of timesteps per sample, m is the
                number of features per timestep and sample and d is the number of
                target variables per sample.

        """
        pass
