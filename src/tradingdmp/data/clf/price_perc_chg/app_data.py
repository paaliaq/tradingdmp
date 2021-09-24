"""Application classes for data pipelines that are used in our trading apps."""

import datetime
from functools import reduce
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.rrule import FR, MO, TH, TU, WE, WEEKLY, rrule
from tradingdmp.data.clf.base_data import BaseFeatureData
from tradingdmp.data.utils.prep_data import PrepData


class DataAlpacaPocCat(BaseFeatureData):
    """Class used for fetching data with categorical target for Alpaca POC.

    The data is daily data from Alphavantage and Finviz for USA tickers.
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

    def _check_inputs(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool,
        return_last_date: bool,
        return_nonlast_dates: bool,
        return_training_dfs: bool,
        return_date_col: bool,
        return_ticker_col: bool,
        bins: List[Any],
        bin_labels: List[str],
        n_ppc_per_row: int,
    ) -> None:
        """Function to check standard inputs across all public data fetch functions."""
        # Type checks
        self._check_inputs_type(
            ticker_list,
            dt_start,
            dt_end,
            dt_end_required,
            return_last_date,
            return_nonlast_dates,
            return_training_dfs,
            return_date_col,
            return_ticker_col,
            bins,
            bin_labels,
            n_ppc_per_row,
        )

        # Logical checks
        self._check_inputs_logic(
            ticker_list,
            dt_start,
            dt_end,
            dt_end_required,
            return_last_date,
            return_nonlast_dates,
            return_training_dfs,
            return_date_col,
            return_ticker_col,
            bins,
            bin_labels,
            n_ppc_per_row,
        )

    def _check_inputs_type(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool,
        return_last_date: bool,
        return_nonlast_dates: bool,
        return_training_dfs: bool,
        return_date_col: bool,
        return_ticker_col: bool,
        bins: List[Any],
        bin_labels: List[str],
        n_ppc_per_row: int,
    ) -> None:  # noqa: C901
        """Auxilary function to check that the input arguments have the correct type."""
        if not isinstance(ticker_list, list):
            raise ValueError("ticker_list must be of type list.")
        if not isinstance(dt_start, datetime.datetime):
            raise ValueError("dt_start must be of type datetime.")
        if not isinstance(dt_end, datetime.datetime):
            raise ValueError("dt_end must be of type datetime.")
        if not isinstance(dt_end_required, bool):
            raise ValueError("dt_end_required must be of type bool.")
        if not isinstance(return_last_date, bool):
            raise ValueError("return_last_date must be of type bool.")
        if not isinstance(return_nonlast_dates, bool):
            raise ValueError("return_nonlast_dates must be of type bool.")
        if not isinstance(return_training_dfs, bool):
            raise ValueError("return_training_dfs must be of type bool.")
        if not isinstance(return_date_col, bool):
            raise ValueError("return_date_col must be of type bool.")
        if not isinstance(return_ticker_col, bool):
            raise ValueError("return_ticker_col must be of type bool.")
        if not isinstance(bins, list):
            raise ValueError("bins must be of type list.")
        if not isinstance(bin_labels, list):
            raise ValueError("bin_labels must be of type list.")
        if not isinstance(n_ppc_per_row, int):
            raise ValueError("n_ppc_per_row must be of type int.")

    def _check_inputs_logic(
        self,
        ticker_list: List[str],
        dt_start: datetime.datetime,
        dt_end: datetime.datetime,
        dt_end_required: bool,
        return_last_date: bool,
        return_nonlast_dates: bool,
        return_training_dfs: bool,
        return_date_col: bool,
        return_ticker_col: bool,
        bins: List[Any],
        bin_labels: List[str],
        n_ppc_per_row: int,
    ) -> None:
        """Auxilary function to check that the input arguments are logically correct.

        This function assumes that all checks from _check_inputs_type passed. This means
        that _check_inputs_type must always be run before _check_inputs_logic.
        """
        if not len(ticker_list) > 0:
            raise ValueError("ticker_list must not be empty.")
        if not len(bins) >= 3:
            raise ValueError("bins must contain at least 3 elements to form 2 classes.")
        if not len(bins) + 1 == len(bin_labels):
            raise ValueError("It is required that: len(bins) == len(bin_labels) + 1")
        timedelta_weekdays = rrule(
            WEEKLY, byweekday=(MO, TU, WE, TH, FR), dtstart=dt_start, until=dt_end
        ).count()  # type: ignore
        if not timedelta_weekdays >= n_ppc_per_row + 2:
            raise ValueError(
                "Number of weekdays btw. dt_start and dt_end must be "
                ">= n_ppc_per_row+2."
            )

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

    # flake8: noqa: C901
    def get_data(
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
        Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], Tuple[pd.DataFrame, pd.DataFrame]
    ]:
        """Method for getting data that can be passed to a model.

        This function fetches pre-processed data from Alphavantage and Finviz,
        merges this data by day and ticker, conducts basic feature engineering,
        constructs the target variable y and then returns the data for x and y.

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
            **kwargs: This is a dictionary which can contain the following keys:
                - n_ppc_per_row (int): An integer defaulting to 10. It represents the
                    minimum number of price percentages changes per row. This mainly
                    affects the price data from Alphavantage, based on which the daily
                    price percentage changes are computed. Important: this affects the
                    number of dates required per ticker. Example: if n_ppc_per_row is
                    10, then there will only be data returned for a particular ticker if
                    this ticker has at least 10+2 dates of data available in the mongodb
                    (+2 asnwe need 1 date in the beginning and end of a sequence of
                    dates to compute percentage changes).

        Returns:
            data: If return_training_dfs is False, the return type is
                Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary that contains
                one key-value pair for each ticker symbol. The key always represents the
                ticker symbol. There is one key for each element in the input
                ticker_list. Each value is a tuple of two pandas data frames x and y:
                x of shape (n, m) and y of shape (n, 1), where n is the number of
                samples, m is the number of features. If return_training_dfs is True,
                the return type is Tuple[pd.DataFrame, pd.DataFrame], a tuple of pandas
                data frames x and y. Both data frames contain the data for all tickers
                and dates combined.
        """
        # Initialize default kwargs if necessary
        if "n_ppc_per_row" not in kwargs:
            n_ppc_per_row = 10
        else:
            n_ppc_per_row = kwargs["n_ppc_per_row"]

        # Check inputs
        self._check_inputs(
            # Non-kwargs
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
            # Kwargs
            n_ppc_per_row=n_ppc_per_row,
        )

        # Adjust other variables
        bins = [-np.inf] + bins + [np.inf]

        # Get data
        df_all = pd.DataFrame()
        df_all_av = self.pdata.usa_alphavantage_eod(ticker_list, dt_start, dt_end)
        df_all_fv = self.pdata.usa_finviz_api(ticker_list, dt_start, dt_end)

        for ticker in ticker_list:

            df_av = df_all_av.loc[df_all_av.ticker == ticker, :]
            df_fv = df_all_fv.loc[df_all_fv.ticker == ticker, :]

            if not df_av.empty and not df_fv.empty:

                # Add prefix for data columns
                idcols = ["ticker", "date"]
                df_av.columns = [c if c in idcols else "av_" + c for c in df_av.columns]
                df_fv.columns = [c if c in idcols else "fv_" + c for c in df_fv.columns]

                # Merge all data sources by date
                df_list = [df_av, df_fv]
                df = reduce(lambda l, r: pd.merge(l, r, on=["ticker", "date"]), df_list)

                # Skip adding df to data_dict if df does not fulfill filter conditions
                if dt_end_required:
                    # Check if the dt_end is available for the ticker.
                    dt_end_str = dt_end.date().strftime("%Y-%m-%d")
                    dt_end_avail = dt_end_str in df.date.astype(str).to_list()
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
        self._check_data(df_all.drop(columns=["y"]))  # exclude y because it can have NA

        # Select rows
        if return_last_date and return_nonlast_dates:
            df_all = df_all.reset_index(drop=True)
        elif return_last_date:
            df_all = df_all.loc[df_all.y.isna(), :].reset_index(drop=True)
        elif return_nonlast_dates:
            df_all = df_all.loc[~df_all.y.isna(), :].reset_index(drop=True)

        # Format output as tuple of data frames or as dict of tuples of data frames
        if return_training_dfs:
            # Tuple of data frames
            df_x = df_all.drop(columns=["y"])
            df_y = df_all.loc[:, "y"].to_frame()

            # Drop ticker and date if necessary
            if not return_ticker_col:
                df_x = df_x.drop(columns=["ticker"])
            if not return_date_col:
                df_x = df_x.drop(columns=["date"])

            # Return result
            return (df_x, df_y)
        else:
            # Dict of tuples of data frames, where keys represent the ticker
            data_dict = dict()
            df_all = df_all.reset_index(drop=True)
            for ticker in df_all.ticker.unique():
                df_x = df_all.loc[df_all.ticker == ticker, :].drop(columns=["y"])
                df_y = df_all.loc[df_all.ticker == ticker, "y"]

                # Drop ticker and date if necessary
                if not return_ticker_col:
                    df_x = df_x.drop(columns=["ticker"])
                if not return_date_col:
                    df_x = df_x.drop(columns=["date"])

                data_dict[ticker] = (df_x, df_y)

            # Return result
            return data_dict
