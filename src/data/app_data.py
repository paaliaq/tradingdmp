"""Application classes for data pipelines that are used in our trading apps."""

import datetime
from functools import reduce
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data.base_data import BaseFeatureData
from data.prep_data import PrepData


class DataAlpacaPocCat(BaseFeatureData):
    """Class used for fetching data from database for Alpaca POC.

    The data is daily data from Alphavantage, Yahoo and Finviz for USA tickers.
    No feature engineering is conducted. Only preprocessing is conducted.
    """

    def __init__(self, mongodbkey: str):
        """Initializes PrepData instance for getting processed data."""
        self.pdata = PrepData(mongodbkey=mongodbkey)

    def _check_data(
        self, data_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> None:
        """Function to check that the data meets all our requirements."""
        for key, value in data_dict.items():

            # Check that there are no NaN or infinite values in df_x
            # We only check df_x and not df_y because df_y can contain NaN because
            # we do not have the price value of "tomorrow" and therefore cannot
            # compute the price percentage changes for the last row of each ticker.
            contains_nan = value[0].isin([np.nan]).any(axis=None)
            if contains_nan:
                raise ValueError(f"df_x contains NaN values for {key}.")
            contains_inf = value[0].isin([np.inf, -np.inf]).any(axis=None)
            if contains_inf:
                raise ValueError(f"df_x contains inf or -inf values for {key}.")

    def get_data(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool = False,
        n_dates_per_row: int = 10,
        return_y: bool = True,
        bins: List[Any] = [-np.inf, -0.03, -0.01, 0.01, 0.03, np.inf],
        bin_labels: List[str] = ["lg_dec", "sm_dec", "no_chg", "sm_inc", "lg_inc"],
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
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
                contain a key value pair for a ticker if there is data available for
                this ticker for the dt_end date.
            n_dates_per_row: Minimum number of dates required for a particular ticker
                symbol. If there are fewer dates available for a ticker, then the
                returned data_dict won't contain a key value pair for this ticker.
            return_y: Whether the data frame with the targets y should be returned. If
                False, the second dataframe in Tuple[pd.DataFrame, pd.DataFrame]] is
                an empty data frame.
            bins: The bins for converting the numeric percentage changes (representing
                the target to be predicted) into a multi-class categorical variable.
            bin_labels: The labels given to the levels of the multi-class categorical
                variable created according to the input argument 'bins'.

        Returns:
            data_dict: a dictionary that contains one key-value pair for each
                ticker symbol. The key always represents the ticker symbol. There is one
                key for each element in the input ticker_list. Each value is a tuple of
                two pandas data frames x and y: x of shape (n, m) and y of shape (n, d),
                where n is the number of samples, m is the number of features and d is
                the number of target variables.
        """
        # Get data
        n_dates_per_row = n_dates_per_row + 2  # 2 more required for perc. changes
        dt_start = dt_start - datetime.timedelta(days=n_dates_per_row)
        data_dict = dict()
        for ticker in ticker_list:

            df_av = self.pdata.usa_alphavantage_eod([ticker], dt_start, dt_end)
            df_yh = self.pdata.usa_yahoo_api([ticker], dt_start, dt_end)
            df_fv = self.pdata.usa_finviz_api([ticker], dt_start, dt_end)

            # DEBUGGING
            print("\n\nTicker: {}".format(ticker))
            print(
                "len(df_av): {} from {} to {}".format(
                    len(df_av), df_av.date.min(), df_av.date.max()
                )
            )
            print(
                "len(df_yh): {} from {} to {}".format(
                    len(df_yh), df_yh.date.min(), df_yh.date.max()
                )
            )
            print(
                "len(df_fv): {} from {} to {}".format(
                    len(df_fv), df_fv.date.min(), df_fv.date.max()
                )
            )

            if not df_av.empty and not df_yh.empty and not df_fv.empty:

                # Add prefix for data columns
                idcols = ["ticker", "date"]
                df_av.columns = [c if c in idcols else "av_" + c for c in df_av.columns]
                df_yh.columns = [c if c in idcols else "yh_" + c for c in df_yh.columns]
                df_fv.columns = [c if c in idcols else "fv_" + c for c in df_fv.columns]

                # Merge all data sources by date
                df_list = [df_av, df_yh, df_fv]
                df = reduce(lambda l, r: pd.merge(l, r, on=["ticker", "date"]), df_list)

                # DEBUGGING
                print(
                    "len(df): {} from {} to {}".format(
                        len(df), df.date.min(), df.date.max()
                    )
                )

                # Skip adding df to data_dict if df does not fulfill filter conditions
                if dt_end_required:
                    # Check if the dt_end is available for the ticker.
                    dt_end_avail = pd.to_datetime(dt_end) in df.date.to_list()
                    if not dt_end_avail:
                        continue

                # If there are not sufficient dates for this ticker, do not return it
                sufficient_dates_avail = n_dates_per_row - 1 <= len(df.date.unique())
                if not sufficient_dates_avail:

                    # DEBUGGING
                    print("sufficient_dates_avail: {}".format(sufficient_dates_avail))
                    continue

                # Create y as percentage change of av_close from current to next day
                avcols = df.columns.str.startswith("av_")
                df_pct = df.loc[:, avcols].replace(0.0, 0.0001).pct_change()
                df.loc[:, "y"] = df_pct.av_close.iloc[1:].to_list() + [np.nan]
                df["y"] = pd.cut(df["y"], bins, labels=bin_labels)

                # Convert OHCL data to df_pct with percentage changes from day to day
                key_cols = range(n_dates_per_row - 2)
                df_pct = pd.concat(
                    [df_pct.shift(-i) for i in key_cols],
                    axis=1,
                    keys=map(str, key_cols),
                ).dropna()
                df_pct.columns = df_pct.columns.map(lambda x: x[1] + "_" + x[0])
                df_pct = df_pct.reset_index(drop=True)

                # Merge df_pct with df_static, the non-percentage change data
                df_static = df.tail(len(df_pct)).reset_index(drop=True)
                df_merged = pd.merge(
                    df_static, df_pct, left_index=True, right_index=True, how="left"
                )

                # Split data into X and y and save in data_dict
                df_x = df_merged.drop(columns=["ticker", "date", "y"])
                df_y = df_merged.loc[:, "y"].to_frame()
                data_dict[ticker] = (df_x, df_y)

        # Check data
        self._check_data(data_dict)

        return data_dict
