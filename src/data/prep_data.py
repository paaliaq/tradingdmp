"""Utility functions to get processed data from our database.

The processing includes the following:
- transformation of dtypes of columns
- sorting by ticker and time
- handling NA values
- filtering columns
- renaming columns
The processing excludes adding or transforming features, which belongs in final_data.py.
"""
import datetime
import gc
from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from data.raw_data import RawData


class InterimData:
    """Class used for fetching processed data from database."""

    def __init__(self, mongodbkey: str):
        """Initializes RawData instance for getting the raw data to be processed."""
        self.rd = RawData(mongodbkey=mongodbkey)

    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to return data sorted by ticker and date with new index."""
        df = df.sort_values(["ticker", "date"])
        df = df.reset_index(drop=True)
        return df

    def _check_data(self, df: pd.DataFrame) -> None:
        """Function to check that the data meets all our requirements."""
        # Check that required columns exist
        if "date" not in df.columns:
            raise ValueError("df must be contain a 'date' column!")
        if "ticker" not in df.columns:
            raise ValueError("df must be contain a 'ticker' column!")

        # Check column datatypes
        if not is_datetime(df.date):
            raise ValueError("date must be of dtype datetime!")

        # We do not check that index is _id because a single _id can turn in man columns
        # for some data sources, such as IEX.

    def usa_alphavantage_eod(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches interim data from alphavantage based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ interimdata = InterimData(mongodbkey)
            $ data = interimdata.usa_alphavantage_eod(ticker_list = ["MSFT", "AAPL"])
        """
        # Get raw data from database
        df = self.rd.usa_alphavantage_eod(ticker_list, dt_start, dt_end)

        # Keep rows with data in 'data' column
        df = df.loc[df.loc[:, "n_cols"] > 0, :]

        # Remove metadata columns
        colnames = ["ticker", "date", "data"]
        df = df.loc[:, colnames]

        # Extract 'data' column into multiple columns
        df = pd.concat([df.drop(["data"], axis=1), df["data"].apply(pd.Series)], axis=1)

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
        """Fetches interim data from IEX based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ interimdata = InterimData(mongodbkey)
            $ data = interimdata.usa_iex_1min(ticker_list = ["MSFT", "AAPL"])
        """
        # Get raw data from database
        df = self.rd.usa_iex_1min(ticker_list, dt_start, dt_end)

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
        """Fetches interim data from yahoo based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ interimdata = InterimData(mongodbkey)
            $ data = interimdata.usa_yahoo_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Get raw data from database
        df = self.rd.usa_yahoo_api(ticker_list, dt_start, dt_end)

        # Keep rows with data in 'data' column
        df = df.loc[df.loc[:, "n_cols"] > 0, :]

        # Remove metadata columns
        colnames = ["ticker", "date", "data"]
        df = df.loc[:, colnames]

        # Extract 'data' column into multiple columns
        df = pd.concat([df.drop(["data"], axis=1), df["data"].apply(pd.Series)], axis=1)

        # Remove data columns to be excluded
        colnames = [
            "currency",
            "displayName",
            "earningsTimestamp",
            "earningsTimestampEnd",
            "earningsTimestampStart",
            "esgPopulated",
            "exchange",
            "exchangeDataDelayedBy",
            "exchangeTimeZone",
            "exchangeTimezoneName",
            "exchangeTimezoneShortName",
            "fiftyTwoWeekRange",
            "financialCurrency",
            "firstTradeDateMilliseconds",
            "fullExchangeName",
            "gmtOffSetMilliseconds",
            "language",
            "longName",
            "market",
            "marketState",
            "messageBoardId",
            "postMarketTime",
            "priceHint",
            "quoteSourceName",
            "quoteType",
            "region",
            "regularMarketDayRange",
            "regularMarketTime",
            "shortName",
            "sourceInterval",
            "tradeable",
            "triggerable",
        ]
        df = df.drop(columns=colnames, errors="ignore")

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
        """Fetches interim data from finviz based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ interimdata = InterimData(mongodbkey)
            $ data = interimdata.usa_finviz_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Get raw data from database
        df = self.rd.usa_finviz_api(ticker_list, dt_start, dt_end)

        # Keep rows with data in 'data' column
        df = df.loc[df.loc[:, "n_cols"] > 0, :]

        # Remove metadata columns
        colnames = ["ticker", "date", "data"]
        df = df.loc[:, colnames]

        # Extract 'data' column into multiple columns
        df = pd.concat([df.drop(["data"], axis=1), df["data"].apply(pd.Series)], axis=1)

        # Remove data columns to be excluded
        colnames = [
            "Company",
            "Country",
            "Index",
            "InstTrans",
            "Earnings",
            "52WRange",
            "Volatility",
        ]
        df = df.drop(columns=colnames, errors="ignore")

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
                    col_new = [float(i[:-1]) * (10 ** m[i[-1]]) for i in df.loc[:, col]]
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
