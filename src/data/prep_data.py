"""Utility functions to get pre-processed data from our database.

The pre-processing includes the following:
- transformation of dtypes of columns
- sorting by ticker and time
- handling NA values
- filtering columns
- renaming columns
The pre-processing excludes adding/transforming features, which belongs in feat_data.py.
"""
import datetime
import gc
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from data.raw_data import RawData


class PrepData:
    """Class used for fetching processed data from database."""

    def __init__(self, mongodbkey: str):
        """Initializes RawData instance for getting the raw data to be processed."""
        self.rd = RawData(mongodbkey=mongodbkey)

    def _check_inputs(
        self,
        ticker_list: List[str],
        dt_start: datetime.date,
        dt_end: datetime.date,
    ) -> None:
        """Function to check standard inputs across all public data fetch functions."""
        # Type checks
        if not isinstance(dt_start, datetime.datetime):
            raise TypeError("dt_start must be of type datetime.")
        if not isinstance(dt_end, datetime.datetime):
            raise TypeError("dt_end must be of type datetime.")
        if not isinstance(ticker_list, list):
            raise TypeError("ticker_list must be of type list.")

        # Logical checks
        if dt_start > dt_end:
            raise ValueError("dt_start must be less than dt_end.")

    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to return data sorted by ticker and date with new index."""
        df = df.sort_values(["ticker", "date"])
        df = df.reset_index(drop=True)
        return df

    def _check_data(self, df: pd.DataFrame) -> None:
        """Function to check that the data meets all our requirements."""
        if not df.empty:
            # Check that required columns exist
            if "date" not in df.columns:
                raise ValueError("df must be contain a 'date' column.")
            if "ticker" not in df.columns:
                raise ValueError("df must be contain a 'ticker' column.")

            # Check column datatypes
            if not is_datetime(df.date):
                raise ValueError("date must be of dtype datetime.")

            # Check that there are no NaN or infinite values
            contains_nan = df.isin([np.nan]).any(axis=None)
            if contains_nan:
                raise ValueError("df must not contain NaN values.")
            contains_inf = df.isin([np.inf, -np.inf]).any(axis=None)
            if contains_inf:
                raise ValueError("df must not contain inf or -inf values.")

        # We do not check that index is _id because a single _id can turn in man columns
        # for some data sources, such as IEX.

    def usa_alphavantage_eod(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches processed data from alphavantage based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ pdata = PrepData(mongodbkey)
            $ data = pdata.usa_alphavantage_eod(ticker_list = ["MSFT", "AAPL"])
        """
        # Get raw data from database
        df = self.rd.usa_alphavantage_eod(ticker_list, dt_start, dt_end)

        if not df.empty:
            # Keep rows with data in 'data' column
            df = df.loc[df.loc[:, "n_cols"] > 0, :]

            # Remove metadata columns
            colnames = ["ticker", "date", "data"]
            df = df.loc[:, colnames]

            # Extract 'data' column into multiple columns
            df = pd.concat(
                [df.drop(["data"], axis=1), df["data"].apply(pd.Series)], axis=1
            )

            # Select required columns
            colnames = [
                "ticker",
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "adjusted_close",
                "dividend_amount",
                "split_coefficient",
            ]
            df = df.loc[:, colnames]

            # Transform dtypes
            df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])

            # Sort by ticker and date and reset index
            df = self._sort_data(df)

            # Replace -9999 with NAN
            df = df.replace(-9999, np.nan)

            # Replace NAN by ticker
            df = df.groupby("ticker").apply(
                lambda group: group.interpolate(method="linear")
            )
            df = df.dropna(axis=0)

        # Check data
        self._check_data(df)

        return df

    def usa_iex_1min(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches processed data from IEX based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ pdata = PrepData(mongodbkey)
            $ data = pdata.usa_iex_1min(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Get raw data from database
        df = self.rd.usa_iex_1min(ticker_list, dt_start, dt_end)

        if not df.empty:
            # Keep rows with data in 'data' column
            df = df.loc[df.loc[:, "n_rows"] > 0, :]

            # Remove metadata columns
            colnames = ["ticker", "date", "data"]
            df = df.loc[:, colnames]

            # Extract 'data' column into multiple columns
            df_all = pd.DataFrame()
            for i, row in df.iterrows():
                df_new = pd.DataFrame(row["data"])
                df_new.loc[:, "ticker"] = row["ticker"]
                df_new.loc[:, "date"] = row["date"]
                df_all = df_all.append(df_new)

            df = df_all.copy()
            del df_all
            gc.collect()

            # Select required columns
            colnames = [
                "ticker",
                "date",
                "high",
                "low",
                "open",
                "close",
                "average",
                "volume",
                "notional",
                "numberOfTrades",
            ]
            df = df.loc[:, colnames]

            # Adjust date column to include the time
            df.loc[:, "date"] = pd.to_datetime(df.date + " " + df.minute)
            df = df.drop(columns=["minute"])

            # Rearrange columns
            colnames = df.columns.to_list()
            colnames.remove("ticker")
            colnames.remove("date")
            df = df.loc[:, ["ticker", "date"] + colnames]

            # Sort by ticker and date and reset index
            df = self._sort_data(df)

            # Replace -9999 with NAN
            df = df.replace(-9999, np.nan)

            # Replace NAN by ticker
            df = df.groupby("ticker").apply(
                lambda group: group.interpolate(method="linear")
            )
            df = df.dropna(axis=0)

        # Check data
        self._check_data(df)

        return df

    def usa_yahoo_api(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches processed data from yahoo based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ pdata = PrepData(mongodbkey)
            $ data = pdata.usa_yahoo_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Get raw data from database
        df = self.rd.usa_yahoo_api(ticker_list, dt_start, dt_end)

        if not df.empty:
            # Keep rows with data in 'data' column
            df = df.loc[df.loc[:, "n_cols"] > 0, :]

            # Remove metadata columns
            colnames = ["ticker", "date", "data"]
            df = df.loc[:, colnames]

            # Extract 'data' column into multiple columns
            df = pd.concat(
                [df.drop(["data"], axis=1), df["data"].apply(pd.Series)], axis=1
            )

            # Select required columns
            colnames = [
                "ticker",
                "date",
                "regularMarketChange",
                "regularMarketChangePercent",
                "regularMarketPrice",
                "regularMarketDayHigh",
                "regularMarketDayLow",
                "regularMarketVolume",
                "regularMarketPreviousClose",
                "bid",
                "ask",
                "bidSize",
                "askSize",
                "regularMarketOpen",
                "averageDailyVolume3Month",
                "averageDailyVolume10Day",
                "fiftyTwoWeekLowChange",
                "fiftyTwoWeekLowChangePercent",
                "fiftyTwoWeekHighChange",
                "fiftyTwoWeekHighChangePercent",
                "fiftyTwoWeekLow",
                "fiftyTwoWeekHigh",
                "sharesOutstanding",
                "fiftyDayAverage",
                "fiftyDayAverageChange",
                "fiftyDayAverageChangePercent",
                "twoHundredDayAverage",
                "twoHundredDayAverageChange",
                "twoHundredDayAverageChangePercent",
                "marketCap",
                "price",
            ]
            df = df.loc[:, colnames]

            # Transform dtypes
            df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])
            colnames = df.columns.to_list()
            colnames.remove("ticker")
            colnames.remove("date")
            for col in colnames:
                try:
                    df.loc[:, col] = df.loc[:, col].astype(float)
                except ValueError:
                    df.loc[:, col] = df.loc[:, col].astype(object)

            # Sort by ticker and date and reset index
            df = self._sort_data(df)

            # Replace -9999 with NAN
            df = df.replace(-9999, np.nan)

            # Replace NAN by ticker
            df = df.groupby("ticker").apply(
                lambda group: group.interpolate(method="linear")
            )
            df = df.dropna(axis=0)

        # Check data
        self._check_data(df)

        return df

    def usa_finviz_api(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches processed data from finviz based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ pdata = PrepData(mongodbkey)
            $ data = pdata.usa_finviz_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Get raw data from database
        df = self.rd.usa_finviz_api(ticker_list, dt_start, dt_end)

        if not df.empty:
            # Keep rows with data in 'data' column
            df = df.loc[df.loc[:, "n_cols"] > 0, :]

            # Remove metadata columns
            colnames = ["ticker", "date", "data"]
            df = df.loc[:, colnames]

            # Extract 'data' column into multiple columns
            df = pd.concat(
                [df.drop(["data"], axis=1), df["data"].apply(pd.Series)], axis=1
            )

            # Select required columns
            colnames = [
                "ticker",
                "date",
                "Sector",
                "Industry",
                "PE",
                "EPSttm",
                "InsiderOwn",
                "ShsOutstand",
                "PerfWeek",
                "MarketCap",
                "ForwardPE",
                "EPSnextY",
                "InsiderTrans",
                "ShsFloat",
                "PerfMonth",
                "Income",
                "PEG",
                "EPSnextQ",
                "InstOwn",
                "ShortFloat",
                "PerfQuarter",
                "Sales",
                "PS",
                "EPSthisY",
                "ShortRatio",
                "PerfHalfY",
                "Booksh",
                "PB",
                "ROA",
                "TargetPrice",
                "PerfYear",
                "Cashsh",
                "PC",
                "EPSnext5Y",
                "ROE",
                "PerfYTD",
                "Dividend",
                "PFCF",
                "EPSpast5Y",
                "ROI",
                "52WHigh",
                "Beta",
                "QuickRatio",
                "Salespast5Y",
                "GrossMargin",
                "52WLow",
                "ATR",
                "Employees",
                "CurrentRatio",
                "SalesQQ",
                "OperMargin",
                "RSI14",
                "Optionable",
                "DebtEq",
                "EPSQQ",
                "ProfitMargin",
                "RelVolume",
                "PrevClose",
                "Shortable",
                "LTDebtEq",
                "Payout",
                "AvgVolume",
                "Price",
                "Recom",
                "SMA20",
                "SMA50",
                "SMA200",
                "Volume",
                "Change",
            ]
            df = df.loc[:, colnames]

            # Format object columns so they can be converted to numeric types
            colnames = df.columns.to_list()
            colnames.remove("ticker")
            colnames.remove("date")

            m = {"K": 3, "M": 6, "B": 9, "T": 12}
            for col in colnames:
                # Remove '%' unit at end of string
                df.loc[:, col] = df.loc[:, col].str.replace("%", "")

                # Remove ','
                df.loc[:, col] = df.loc[:, col].str.replace(",", "")

                # Convert endings with K, M, B, T into numeric form
                if (
                    df.loc[:, col].str.endswith("K").any()
                    or df.loc[:, col].str.endswith("M").any()
                    or df.loc[:, col].str.endswith("B").any()
                    or df.loc[:, col].str.endswith("T").any()
                ):
                    try:
                        col_new = [
                            float(i[:-1]) * (10 ** m[i[-1]]) for i in df.loc[:, col]
                        ]
                        df.loc[:, col] = [round(x) for x in col_new]
                    except ValueError:
                        pass

            # Transform dtypes
            df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])
            for col in colnames:
                try:
                    df.loc[:, col] = df.loc[:, col].astype(float)
                except ValueError:
                    df.loc[:, col] = df.loc[:, col].astype(object)

            # Sort by ticker and date and reset index
            df = self._sort_data(df)

            # Replace -9999 with NAN
            df = df.replace(-9999, np.nan)

            # Replace NAN by ticker
            df = df.groupby("ticker").apply(
                lambda group: group.interpolate(method="linear")
            )
            df = df.dropna(axis=0)

        # Check data
        self._check_data(df)

        return df
