import datetime
from functools import reduce
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tradingdmp.data.reg.base_data import BaseFeatureData
from tradingdmp.data.utils.prep_data import PrepData


class DataIBPocReg(BaseFeatureData):
    """Class used for fetching data with categorical target for Alpaca POC.

    The data is daily data from Alphavantage, Yahoo and Finviz for USA tickers.
    The target y consists of discretized daily price percentage changes (from the
    previous day closing price to the next day closing price). Thereby, discretization
    is conducted as defined by the get_data function arguments 'bins' and 'bin_labels'.
    Feature engineering is only conducted in very limited way: OCHL time series from
    Alphavantage are converted to features by computing daily price percentage changes
    and then adding the last `n_ppc_per_row` price percentage changes as new columns.
    """

    def __init__(self, mongodbkey: str):
        """Initializes PrepData instance for getting processed data."""
        self.pdata = PrepData(mongodbkey=mongodbkey)

    def _check_inputs_type(self, ticker_list: List[str],
                           dt_start: datetime.datetime,
                           dt_end: datetime.datetime) -> None:  # noqa: C901
        """Auxilary function to check that the input arguments have the correct type."""
        if not isinstance(ticker_list, list):
            raise ValueError("ticker_list must be of type list.")
        if not isinstance(dt_start, datetime.datetime):
            raise ValueError("dt_start must be of type datetime.")
        if not isinstance(dt_end, datetime.datetime):
            raise ValueError("dt_end must be of type datetime.")

    def _check_data(self, df_x: pd.DataFrame) -> None:
        """Function to check that the data meets all our requirements."""
        # Check that there are no NaN or infinite values in df_x
        # We only check df_x and not df_y because df_y can contain NaN because
        # we do not have the price value of "tomorrow" and therefore cannot
        # compute the price percentage changes for the last row of each ticker.
        contains_nan = df_x.isin([np.nan]).any(axis=None)
        if contains_nan:
            raise ValueError("df_x contains NaN values.")
        contains_inf = df_x.isin([np.inf, -np.inf]).any(axis=None)
        if contains_inf:
            raise ValueError("df_x contains inf or -inf values.")

    def get_data(self,
                 ticker_list: List[str],
                 dt_start: datetime.datetime,
                 dt_end: datetime.datetime,
                 history_len_daily: int = 10,
                 history_len_quarterly: int = 4,
                 history_len_yearly: int = 2,
                 **kwargs: Any) -> pd.DataFrame:
        """Method for getting data that can be passed to a model.

        This function fetches pre-processed data from bors-data, Yahoo and Finviz,
        merges this data by day and ticker, conducts basic feature engineering,
        constructs the target variable y and then returns the data for x and y.

        Args:
            ticker_list: A list of ticker symbols for which to get data.
            dt_start: All data after incl. dt_start is fetched.
            dt_end: All data until incl. dt_end is fetched.
            **kwargs

        Returns:
            data: If return_training_dfs is False, the return type is

        """

        self._check_inputs_type(ticker_list, dt_start, dt_end)

        data_daily = self.pdata.bors_data(ticker_list=ticker_list,
                                          dt_start=dt_start,
                                          dt_end=dt_end,
                                          granularity="daily")

        data_quarterly = self.pdata.bors_data(ticker_list=ticker_list,
                                              dt_start=dt_start,
                                              dt_end=dt_end,
                                              granularity="quarterly")

        data_yearly = self.pdata.bors_data(ticker_list=ticker_list,
                                           dt_start=dt_start,
                                           dt_end=dt_end,
                                           granularity="yearly")

        # Process each ticker
        iter_count = 0
        for ticker in ticker_list:

            daily_iter = data_daily.loc[data_daily["ticker"] == ticker]

            quarterly_iter = data_quarterly.loc[data_quarterly["ticker"] ==
                                                ticker]

            yearly_iter = data_yearly.loc[data_yearly["ticker"] == ticker]

            # Processing daily data
            # Convert to numpy array for simple reshaping
            daily_iter_arr = np.array(daily_iter)

            # Create numpy array with daily data features, shape: (observations, (history_len_daily*features))
            daily_feature_array = np.array([
                np.stack(daily_iter_arr)[iter:iter + history_len_daily]
                for iter in range(0, daily_iter.shape[0] - history_len_daily +
                                  1)
            ])

            # Extract target variable, shape: (observations-history_len_daily, )
            daily_tar = np.array(daily_iter["Close"])
            target_array_reshaped = daily_tar[(
                                                  history_len_daily):daily_iter.shape[
                                                                         0] + 1]

            # Create report publication variable
            # What date was the report actually reported not which fiscal period it describes
            quarterly_iter["report_pub_date"] = quarterly_iter.date + \
                                                pd.DateOffset(months=3)
            yearly_iter["report_pub_date"] = yearly_iter.date + \
                                             pd.DateOffset(months=3)

            # Creates the quarterly feature array, shape: (history_len_quarterly, quarter features, observations)
            quarterly_feature_array = self._y_q_df_to_arr(
                daily_feature_array, quarterly_iter, history_len_quarterly)

            # Creates the yearly feature array, shape: (history_len_yearly, yearly features, observations)
            yearly_feature_array = self._y_q_df_to_arr(daily_feature_array,
                                                       yearly_iter,
                                                       history_len_yearly)

            # Create transposed quarterly dataframe for each observation in the ticker
            d_df_iter = self._array_to_df(daily_feature_array,
                                          history_len_daily, "daily",
                                          daily_iter)
            q_df_iter = self._array_to_df(quarterly_feature_array,
                                          history_len_quarterly, "quarterly",
                                          quarterly_iter)

            y_df_iter = self._array_to_df(yearly_feature_array,
                                          history_len_yearly, "yearly",
                                          yearly_iter)

            daily_iter = daily_iter.iloc[history_len_daily:, ]
            daily_iter.reset_index(drop=True, inplace=True)
            d_df_iter.reset_index(drop=True, inplace=True)
            q_df_iter.reset_index(drop=True, inplace=True)
            y_df_iter.reset_index(drop=True, inplace=True)

            full_df_iter = pd.concat([
                d_df_iter, q_df_iter, y_df_iter,
                daily_iter[["date", "ticker", "Close"]]
            ],
                axis=1)

            if iter_count == 0:
                complete_df = full_df_iter
            else:
                # complete_df2 = full_df_iter
                complete_df = pd.concat([complete_df, full_df_iter], join="inner")

            iter_count = +1

        # Check data
        self._check_data(complete_df[complete_df.columns.difference(
            ["date", "ticker", "Close"])])  # exclude y because it can have NA

        return complete_df

    def _array_to_df(self, np_array, seq_len, time_type, df):
        if time_type == "daily":
            np_array = np.moveaxis(np_array, [0, 1, 2], [2, 0, 1])
        for x in range(0, np_array.shape[2]):
            temp_df = pd.DataFrame(data=np_array[:, :, x], columns=df.columns)
            temp_df = temp_df[temp_df.columns.difference(['ticker'])]
            if temp_df["date"].values.dtype == '<M8[ns]':
                temp_df["date"] = time_type + "_" + \
                                  temp_df["date"].rank(
                                      ascending=False).astype(int).astype(str)
            else:
                temp_df["date"] = list(reversed(range(1, seq_len +
                                                      1)))  # if missing data
                temp_df["date"] = time_type + "_" + temp_df["date"].astype(str)
            temp_df = temp_df.set_index(['date']).unstack()
            temp_df.index = [
                '_'.join(map(str, i)) for i in temp_df.index.tolist()
            ]
            temp_df.index = temp_df.index.str.lower().str.replace(
                "-", "_").str.split(" ").str[0]
            temp_df = pd.DataFrame(temp_df).transpose()
            if x == 0:
                df_iter = temp_df
            else:
                df_iter = pd.concat([df_iter, temp_df], axis=0)
        return df_iter

    def _y_q_df_to_arr(self, daily_feature_array, df, seq_len):
        # Creates the quarterly feature array, shape: (history_len_quarterly, quarter features, observations)
        iter_list = []
        for x in range(0, daily_feature_array.shape[0]):
            # if daily date is before quarterly report pub date set that observations filled with zeroes/skip
            q_rows_match = df[
                df["report_pub_date"] <= daily_feature_array[x][-1][0]].index
            if len(q_rows_match) < seq_len:
                feature_array = np.zeros((seq_len, df.shape[1]))
            # Add the last seq_len quarters of data to feature array
            else:
                feature_array = np.array(df.loc[q_rows_match].tail(seq_len))

            iter_list.append(feature_array)
        feature_array = np.dstack(iter_list)
        return feature_array