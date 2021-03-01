"""Application classes for data pipelines that are used in our trading apps."""

import datetime
from functools import reduce
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.rrule import FR, MO, TH, TU, WE, WEEKLY, rrule

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

    def _check_inputs(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool,
        n_ppc_per_row: int,
        return_last_date_only: bool,
        return_training_dfs: bool,
        bins: List[Any],
        bin_labels: List[str],
    ) -> None:
        """Function to check standard inputs across all public data fetch functions."""
        # Type checks
        self._check_inputs_type(
            ticker_list,
            dt_start,
            dt_end,
            dt_end_required,
            n_ppc_per_row,
            return_last_date_only,
            return_training_dfs,
            bins,
            bin_labels,
        )

        # Logical checks
        self._check_inputs_logic(
            ticker_list,
            dt_start,
            dt_end,
            dt_end_required,
            n_ppc_per_row,
            return_last_date_only,
            return_training_dfs,
            bins,
            bin_labels,
        )

        # Logical checks

    def _check_inputs_type(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool,
        n_ppc_per_row: int,
        return_last_date_only: bool,
        return_training_dfs: bool,
        bins: List[Any],
        bin_labels: List[str],
    ) -> None:
        """Auxilary function to check that the input arguments have the correct type."""
        if not isinstance(ticker_list, list):
            raise ValueError("ticker_list must be of type list.")
        if not isinstance(dt_start, datetime.datetime):
            raise ValueError("dt_start must be of type datetime.")
        if not isinstance(dt_end, datetime.datetime):
            raise ValueError("dt_end must be of type datetime.")
        if not isinstance(dt_end_required, bool):
            raise ValueError("dt_end_required must be of type bool.")
        if not isinstance(n_ppc_per_row, int):
            raise ValueError("n_ppc_per_row must be of type int.")
        if not isinstance(return_last_date_only, bool):
            raise ValueError("return_last_date_only must be of type bool.")
        if not isinstance(return_training_dfs, bool):
            raise ValueError("return_training_dfs must be of type bool.")
        if not isinstance(bins, list):
            raise ValueError("bins must be of type list.")
        if not isinstance(bin_labels, list):
            raise ValueError("bin_labels must be of type list.")

    def _check_inputs_logic(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool,
        n_ppc_per_row: int,
        return_last_date_only: bool,
        return_training_dfs: bool,
        bins: List[Any],
        bin_labels: List[str],
    ) -> None:
        """Auxilary function to check that the input arguments are logically correct."""
        if not len(ticker_list) > 0:
            raise ValueError("ticker_list must not be empty.")
        timedelta_weekdays = rrule(
            WEEKLY, byweekday=(MO, TU, WE, TH, FR), dtstart=dt_start, until=dt_end
        ).count()  # type: ignore
        if not timedelta_weekdays >= n_ppc_per_row + 2:
            raise ValueError(
                "No. of weekdays btw. dt_start and dt_end must be >= n_ppc_per_row+2."
            )
        if not len(bins) >= 3:
            raise ValueError("bins must contain at least 3 elements to form 2 classes.")
        if not len(bins) == len(bin_labels) + 1:
            raise ValueError("It is required that: len(bins) == len(bin_labels) + 1")

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

    def get_data(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool = False,
        n_ppc_per_row: int = 10,
        return_last_date_only: bool = False,
        return_training_dfs: bool = False,
        bins: List[Any] = [-np.inf, -0.03, -0.01, 0.01, 0.03, np.inf],
        bin_labels: List[str] = ["lg_dec", "sm_dec", "no_chg", "sm_inc", "lg_inc"],
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
                this ticker for the dt_end date.
            n_ppc_per_row: Minimum number of price percentages changes per row. This
                mainly affects the price data from alphavantage, based on which the
                price percentage changes are computed (from day to day). Important:
                this affects the number of dates required per ticker. Example: if
                n_ppc_per_row is 10, then we need 10+2 dates (+2 because we need 1 date
                in the beginning and end of a sequence of dates to compute percentage
                changes).
            return_last_date_only: Whether only data for the most recent available date
                per ticker should be returned. If this is set to True, then return_y
                is automatically set to False, i.e. y is never returned (since we do not
                know the price percentage change from the last available date to the
                next future date). You should set return_last_date_only to true when
                making predictions during trading.
            return_training_dfs: Whether data should be returned for model fitting or
                not. You will want to set return_training_dfs to True if you need a
                dataset for model training, validation and testing. If set to True, the
                data for all tickers is returned as tuple of data frames: (df_x, df_y).
                You won't know, which row corresponds to which ticker (and date).
                Moreover, rows with NA values for y will be dropped (i.e. the very last
                row for each ticker will be dropped). If set to False, the data for all
                tickers is returned as dictionary of tuples (df_x, df_y), where each key
                value pair corresponds to a particular ticker symbol.
            bins: The bins for converting the numeric price percentage changes (the
                target to be predicted) into a multi-class categorical variable.
            bin_labels: The labels given to the levels of the multi-class categorical
                variable created according to the input argument 'bins'.

        Returns:
            data_dict: If return_training_dfs is False, the return type is
                Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary that contains
                one key-value pair for each ticker symbol. The key always represents the
                ticker symbol. There is one key for each element in the input
                ticker_list. Each value is a tuple of two pandas data frames x and y:
                x of shape (n, m) and y of shape (n, d), where n is the number of
                samples, m is the number of features and d is the number of target
                variables.
                If return_training_dfs is True, the return type is
                Tuple[pd.DataFrame, pd.DataFrame], a type of pandas data frames x and y.
                Both data frames contain the data for all tickers and dates combined.
        """
        # Check inputs
        self._check_inputs(
            ticker_list=ticker_list,
            dt_start=dt_start,
            dt_end=dt_end,
            dt_end_required=dt_end_required,
            n_ppc_per_row=n_ppc_per_row,
            return_last_date_only=return_last_date_only,
            return_training_dfs=return_training_dfs,
            bins=bins,
            bin_labels=bin_labels,
        )

        # Get data
        df_all = pd.DataFrame()

        for ticker in ticker_list:

            df_av = self.pdata.usa_alphavantage_eod([ticker], dt_start, dt_end)
            df_yh = self.pdata.usa_yahoo_api([ticker], dt_start, dt_end)
            df_fv = self.pdata.usa_finviz_api([ticker], dt_start, dt_end)

            if not df_av.empty and not df_yh.empty and not df_fv.empty:

                # Add prefix for data columns
                idcols = ["ticker", "date"]
                df_av.columns = [c if c in idcols else "av_" + c for c in df_av.columns]
                df_yh.columns = [c if c in idcols else "yh_" + c for c in df_yh.columns]
                df_fv.columns = [c if c in idcols else "fv_" + c for c in df_fv.columns]

                # Merge all data sources by date
                df_list = [df_av, df_yh, df_fv]
                df = reduce(lambda l, r: pd.merge(l, r, on=["ticker", "date"]), df_list)

                # Skip adding df to data_dict if df does not fulfill filter conditions
                if dt_end_required:
                    # Check if the dt_end is available for the ticker.
                    dt_end_avail = pd.to_datetime(dt_end) in df.date.to_list()
                    if not dt_end_avail:
                        continue

                # If there are not sufficient dates for this ticker, do not return it
                sufficient_dates_avail = n_ppc_per_row + 1 <= len(df.date.unique())
                if not sufficient_dates_avail:
                    continue

                # Create y as percentage change of av_close from current to next day
                avcols = df.columns.str.startswith("av_")
                df_pct = df.loc[:, avcols].replace(0.0, 0.0001).pct_change()
                df.loc[:, "y"] = df_pct.av_close.iloc[1:].to_list() + [np.nan]
                df["y"] = pd.cut(df["y"], bins, labels=bin_labels)

                # Convert OHCL data to df_pct with percentage changes from day to day
                key_cols = range(n_ppc_per_row)
                df_pct = pd.concat(
                    [df_pct.shift(-i) for i in key_cols],
                    axis=1,
                    keys=map(str, key_cols),
                ).dropna()
                df_pct.columns = df_pct.columns.map(lambda x: x[1] + "_" + x[0])
                df_pct = df_pct.reset_index(drop=True)

                # Merge df_pct with df_static, the non-percentage change data
                df_static = df.tail(len(df_pct)).reset_index(drop=True)
                df_ticker = pd.merge(
                    df_static, df_pct, left_index=True, right_index=True, how="left"
                )
                df_all = df_all.append(df_ticker)

        # Check data
        x_cols = df_all.drop(columns=["ticker", "date", "y"]).columns
        self._check_data(df_all.loc[:, x_cols])

        # Format output as tuple of data frames or as dict of tuples of data frames
        if return_training_dfs:
            # Tuple of data frames
            df_all = df_all.loc[~df_all.y.isna(), :].reset_index(drop=True)
            df_x = df_all.drop(columns=["ticker", "date", "y"])
            df_y = df_all.loc[:, "y"].to_frame()

            # Return result
            return (df_x, df_y)
        else:
            # Dict of tuples of data frames, where keys represent the ticker
            data_dict = dict()
            df_all = df_all.reset_index(drop=True)
            for ticker in df_all.ticker.unique():
                df_x = df_all.loc[df_all.ticker == ticker, x_cols]
                df_y = df_all.loc[df_all.ticker == ticker, "y"]
                data_dict[ticker] = (df_x, df_y)

            # Return result
            return data_dict
