"""Utility functions to get raw, unprocessed data from our database."""
import datetime
import logging
from asyncio import gather
from typing import Any, List, Union

import motor.motor_asyncio as motor
import pandas as pd

logger = logging.getLogger(__name__)


class RawData:
    """Class used for fetching raw data from database."""

    def __init__(self, mongodbkey: str):
        """Initializes connection to mongodb."""
        self.client = motor.AsyncIOMotorClient(mongodbkey)

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

    def _adjust_inputs(
        self,
        ticker_list: List[str],
    ) -> List[str]:
        """Function to adjust standard inputs across all public data fetch functions."""
        # Remove duplicates from list
        ticker_set = set(ticker_list)
        ticker_list = list(ticker_set)

        return ticker_list

    async def _get_ticker(
        self, ticker: str, condition: dict, db: Any
    ) -> Union[pd.DataFrame, None]:

        try:
            cursor = db[ticker].find(condition)
            documents = [document for document in await cursor.to_list(length=100)]
            df_ticker = pd.DataFrame(documents)
            df_ticker.loc[:, "ticker"] = ticker
            logger.warning(f"SUCCESS {ticker}", {ticker: ticker})
            return df_ticker
        except Exception:
            logger.exception(f"Could not query ticker {ticker}", {ticker: ticker})
            return None

    async def _get_data(
        self,
        db: Any,
        condition: dict,
        ticker_list: List[str],
    ) -> pd.DataFrame:
        """Function to get data for all tickers in ticker_list from the specified db."""
        # Kick of all tasks and await their result
        ticker_dfs_raw = await gather(
            *[self._get_ticker(ticker, condition, db) for ticker in ticker_list]
        )

        # Failed taks return "None" and must hence be filtered
        ticker_dfs_ok = filter(lambda df: df is not None, ticker_dfs_raw)

        # Create the data frame
        df = pd.concat(ticker_dfs_ok)

        # Set index
        if not df.empty:
            df.set_index("_id", inplace=True)

        return df

    def _check_data(self, df: pd.DataFrame) -> None:
        """Function to check that the data meets all our requirements."""
        if not df.empty:
            # Check that required columns exist
            if "date" not in df.columns:
                raise ValueError("df must be contain a 'date' column.")
            if "ticker" not in df.columns:
                raise ValueError("df must be contain a 'ticker' column.")

            # Check that index is _id
            if df.index.name != "_id":
                raise ValueError("df index must be '_id'.")

    async def usa_alphavantage_eod(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from alphavantage based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ rdata = RawData(mongodbkey)
            $ data = rdata.usa_alphavantage_eod(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list = self._adjust_inputs(ticker_list)

        # Define function specific constants
        db_name = "USA-alphavantage-eod"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y%m%d"),
                "$lte": dt_end.strftime("%Y%m%d"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = await self._get_data(db, condition, ticker_list)

        # Check data
        self._check_data(df)

        return df

    def usa_iex_1min(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from IEX based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ rdata = RawData(mongodbkey)
            $ data = rdata.usa_iex_1min(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list = self._adjust_inputs(ticker_list)

        # Define function specific constants
        db_name = "USA-IEX-1min"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y%m%d"),
                "$lte": dt_end.strftime("%Y%m%d"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = self._get_data(db, condition, ticker_list)

        # Check data
        self._check_data(df)

        return df

    def usa_yahoo_api(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from yahoo based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ rdata = RawData(mongodbkey)
            $ data = rdata.usa_yahoo_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list = self._adjust_inputs(ticker_list)

        # Define function specific constants
        db_name = "USA-yahoo-api"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y%m%d"),
                "$lte": dt_end.strftime("%Y%m%d"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = self._get_data(db, condition, ticker_list)

        # Check data
        self._check_data(df)

        return df

    async def usa_finviz_api(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from finviz based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ rdata = RawData(mongodbkey)
            $ data = rdata.usa_finviz_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list = self._adjust_inputs(ticker_list)

        # Define function specific constants
        db_name = "USA-finviz-api"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y%m%d"),
                "$lte": dt_end.strftime("%Y%m%d"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = await self._get_data(db, condition, ticker_list)

        # Check data
        self._check_data(df)

        return df

    async def bors_data(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
        granularity: str = "yearly",
    ) -> pd.DataFrame:
        """Fetches raw data from bors-data based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ rd = RawData(mongodbkey)
            $ data = rd.bors_data(ticker_list = ["ABB", "AAB"])
        """
        # Adjust inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        if granularity not in ["yearly", "quarterly", "daily"]:
            raise ValueError(
                "granularity must any of: 'daily', 'quarterly' or 'yearly'"
            )

        # Remove duplicates from list
        ticker_list = self._adjust_inputs(ticker_list)

        # Granularity effect and check
        granularity_dict = {
            "yearly": {
                "collection_name": "bors-data-yearly",
                "time_key": "report_End_Date",
            },
            "quarterly": {
                "collection_name": "bors-data-quarterly",
                "time_key": "report_End_Date",
            },
            "daily": {"collection_name": "bors-data-daily", "time_key": "Time"},
        }

        # Fetch data
        time_key = granularity_dict[granularity]["time_key"]
        condition = {time_key: {"$gte": dt_start, "$lte": dt_end}}
        db = self.client[granularity_dict[granularity]["collection_name"]]
        df = await self._get_data(db, condition, ticker_list)

        df.rename({time_key: "date"}, axis=1, inplace=True)

        # Check data
        self._check_data(df)

        return df
