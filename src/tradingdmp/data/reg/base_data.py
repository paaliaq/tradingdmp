"""PLACEHOLDER: Base classes for data pipelines."""

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
        dt_end_required: bool,
        return_last_date_only: bool,
        return_training_dfs: bool,
        return_date_col: bool,
        return_ticker_col: bool,
        bins: List[Any],
        bin_labels: List[str],
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
            dt_end: All data until incl. dt_end is fetched.
            dt_end_required: Whether data for dt_end is required for a particular ticker
                symbol. If dt_end_required is true, the returned data_dict will only
                contain a key-value pair for a ticker if there is data available for
                this ticker for the dt_end date. Note: if you set return_training_dfs to
                True and return_date_col to True, then you won't actually get any data
                points for the dt_end. This is because the last observation is dropped
                because it only has NA for y and is therefore not useful for training.
            return_last_date_only: Whether only data for the most recent available date
                per ticker should be returned. If this is set to True, then return_y
                is automatically set to False, i.e. y is never returned (since we do not
                know the price percentage change from the last available date to the
                next future date). You should set return_last_date_only to true when
                making predictions during trading.
            return_training_dfs: Whether data should be returned for model training or
                not. You will want to set return_training_dfs to True if you need a
                dataset for model training, validation and testing. If set to True, the
                data for all tickers is returned as tuple of data frames: (df_x, df_y).
                Rows with NA values for y will be dropped (i.e. the very last row for
                each ticker will be dropped). If set to False, the data for all tickers
                is returned as dictionary of tuples (df_x, df_y), where each key-value
                pair corresponds to a particular ticker symbol.
            return_date_col: Whether or not the date column should be kept in df_x.
            return_ticker_col: Whether or not the ticker column should be kept in df_x.
            bins: The bins for converting the original continuous target (e.g. price
                percentage change) into a discrete target y (with at least 2 levels).
            bin_labels: The labels given to the levels of the discrete target y, which
                was created according to the input argument 'bins'.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            data: There are two options. Option A) must be implemented by default.
                Option B) should be implemented for convenience to get the data in a
                format that is suitable for model fitting.
                - Option A): If return_training_dfs is False, the return type is
                Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]. This is a dictionary that
                contains a key-value pair for each ticker symbol. The key always
                represents the ticker symbol. (If there is data for each requested
                ticker, there will be one key for each element in the input ticker_list.
                If there is not enough data for a requested ticker, however, then this
                ticker won't appear in the returned dictionary.) Each value is a tuple
                of two pandas data frames x and y: x of shape (n, m) and y of shape
                (n, d), where n is the number of samples, m is the number of features
                and d is the number of target variables.
                Option B): If return_training_dfs is False, the return type is
                Tuple[pd.DataFrame, pd.DataFrame]. This is a tuple of two data frames
                x and y. These data frames contain all data points for all dates and
                tickers.
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
