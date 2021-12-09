"""Base classes for data pipelines."""

import datetime
import os
import pickle
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

    async def get_data_cached(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool = False,
        return_last_date: bool = False,
        return_nonlast_dates: bool = True,
        return_training_dfs: bool = False,
        return_date_col: bool = False,
        return_ticker_col: bool = False,
        bins: List[Any] = [-0.03, -0.01, 0.01, 0.03],
        bin_labels: List[str] = [
            "0_lg_dec",
            "1_sm_dec",
            "2_no_chg",
            "3_sm_inc",
            "4_lg_inc",
        ],
        **kwargs: Any
    ) -> Union[
        Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        Tuple[pd.DataFrame, pd.DataFrame],
    ]:
        """Method for getting data that can be passed to a model.

        This function should get raw data, clean this data, conduct feature
        engineering and split the data into predictors x and targets y. This function
        should return feature data.
        This function loads data from a local pickle file if available (instead of
        fetching it from mongodb). It fetches data from mongodb and writes it to a
        local pickle file if such a pickle file is not available.
        You should only use this function for training when you always fetch the same
        data without changing the input arguments. If you keep changing inputs, you will
        prefer to use get_data instead. Important: If input arguments change and a
        previously written out file with data is available, then this out-dated data
        file will be loaded (although it was fetched based on older input arguments).

        Args:
            ticker_list: A list of ticker symbols for which to get data.
            dt_start: All data after incl. dt_start is fetched.
            dt_end: All data until incl. dt_end is fetched.
            dt_end_required: Whether data for dt_end is required for a particular ticker
                symbol. If dt_end_required is true, the returned data_dict will only
                contain a key-value pair for a ticker if there is data available for
                this ticker for the dt_end date.
            return_last_date: Whether data for the most recent available date
                per ticker should be returned. This data will have an NA value for the y
                since the future value is not given yet. You should set return_last_date
                to true if you want to make predictions during trading and you should
                set it to false when you want to train a model in order to filter out
                rows with missing y.
            return_nonlast_dates: Whether data for the date before the most recent
                available date per ticker should be returned. This data will have valid
                y values for each observation (i.e. no NA). You should set
                return_nonlast_dates to true if you want to train a model and to false
                when you just want to make predictions during trading.
            return_training_dfs: Whether all data should be returned as a single data
                frame or not. You will want to set return_training_dfs to True if you
                need a data set for model training, validation and testing.
                If set to True, the data for all tickers is returned as tuple of data
                frames: (df_x, df_y). If set to False, the data for all tickers
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
        filepath = "cached_data.pkl"

        # Check if cached data is available
        cached_data_exists = os.path.exists(filepath)

        # Load cached data if available
        if cached_data_exists:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                return data
        # Get and cach data from mongodb if not available
        else:
            data = await self.get_data(
                ticker_list=ticker_list,
                dt_start=dt_start,
                dt_end=dt_end,
                dt_end_required=dt_end_required,
                return_last_date=return_last_date,
                return_nonlast_dates=return_nonlast_dates,
                return_training_dfs=return_training_dfs,
                return_date_col=return_date_col,
                return_ticker_col=return_ticker_col,
                bins=bins,
                bin_labels=bin_labels,
                **kwargs
            )
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            return data

    @abstractmethod
    async def get_data(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool = False,
        return_last_date: bool = False,
        return_nonlast_dates: bool = True,
        return_training_dfs: bool = False,
        return_date_col: bool = False,
        return_ticker_col: bool = False,
        bins: List[Any] = [-0.03, -0.01, 0.01, 0.03],
        bin_labels: List[str] = [
            "0_lg_dec",
            "1_sm_dec",
            "2_no_chg",
            "3_sm_inc",
            "4_lg_inc",
        ],
        **kwargs: Any
    ) -> Union[
        Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        Tuple[pd.DataFrame, pd.DataFrame],
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
            return_last_date: Whether data for the most recent available date
                per ticker should be returned. This data will have an NA value for the y
                since the future value is not given yet. You should set return_last_date
                to true if you want to make predictions during trading and you should
                set it to false when you want to train a model in order to filter out
                rows with missing y.
            return_nonlast_dates: Whether data for the date before the most recent
                available date per ticker should be returned. This data will have valid
                y values for each observation (i.e. no NA). You should set
                return_nonlast_dates to true if you want to train a model and to false
                when you just want to make predictions during trading.
            return_training_dfs: Whether all data should be returned as a single data
                frame or not. You will want to set return_training_dfs to True if you
                need a data set for model training, validation and testing.
                If set to True, the data for all tickers is returned as tuple of data
                frames: (df_x, df_y). If set to False, the data for all tickers
                is returned as dictionary of tuples (df_x, df_y), where each key-value
                pair corresponds to a particular ticker symbol.
            return_date_col: Whether or not the date column should be kept in df_x.
            return_ticker_col: Whether or not the ticker column should be kept in df_x.
            bins: The bins for converting the original continuous target (e.g. price
                percentage change) into a discrete target y (with at least 2 levels).
            bin_labels: The labels given to the levels of the discrete target y, which
                was created according to the input argument 'bins'
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

    async def get_data_cached(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        **kwargs: Any
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Method for getting data that can be passed to a model.

        This function should fetch raw data, clean this data, conduct feature
        engineering and split the data into predictors x and targets y. This function
        should return timeseries data.
        This function loads data from a local pickle file if available (instead of
        fetching it from mongodb). It fetches data from mongodb and writes it to a
        local pickle file if such a pickle file is not available.
        You should only use this function for training when you always fetch the same
        data without changing the input arguments. If you keep changing inputs, you will
        prefer to use get_data instead. Important: If input arguments change and a
        previously written out file with data is available, then this out-dated data
        file will be loaded (although it was fetched based on older input arguments).

        Args:
            ticker_list: A list of ticker symbols for which to get data.
            dt_start: All data after incl. dt_start is fetched.
                By default, dt_start is None, which means that data is fetched
                from the first available datetime onward.
            dt_end: All data until incl. dt_end is fetched.
                By default, dt_end is None, which means that data is fetched
                until the last the available datetime.
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
        filepath = "cached_data.pkl"

        # Check if cached data is available
        cached_data_exists = os.path.exists(filepath)

        # Load cached data if available
        if cached_data_exists:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                return data
        # Get and cach data from mongodb if not available
        else:
            data = await self.get_data(
                ticker_list=ticker_list, dt_start=dt_start, dt_end=dt_end, kwargs=kwargs
            )
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            return data

    @abstractmethod
    async def get_data(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
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
