"""utility function used for DMP."""
import datetime
from typing import List, Tuple

import pandas as pd
import pymongo


class RawData:
    """Class used for fetching raw data from database."""

    def __init__(self, mongodbkey: str):
        """Initializes connection to mongodb."""
        self.client = pymongo.MongoClient(mongodbkey)

    def _check_inputs(
        self,
        ticker_list: List[str],
        dt_start: datetime.date,
        dt_end: datetime.date,
    ) -> None:
        """Function to check standard inputs across all public data fetch functions."""
        # Type checks
        if not isinstance(dt_start, datetime.datetime):
            raise TypeError("dt_start must be of type datetime!")

        if not isinstance(dt_end, datetime.datetime):
            raise TypeError("dt_end must be of type datetime!")

        if not isinstance(ticker_list, list):
            raise TypeError("ticker_list must be of type list!")

        # Logical checks
        if dt_start > dt_end:
            raise ValueError("dt_start must be less than dt_end!")

    def _adjust_inputs(
        self,
        ticker_list: List[str],
        dt_start: datetime.date,
        dt_end: datetime.date,
    ) -> Tuple[List[str], datetime.date, datetime.date]:
        """Function to adjust standard inputs across all public data fetch functions."""
        # Remove duplicates from list
        ticker_set = set(ticker_list)
        ticker_list = list(ticker_set)

        return (ticker_list, dt_start, dt_end)

    def _get_data(
        self,
        db: pymongo.database.Database,
        condition: dict,
        ticker_list: List[str],
        dt_start: datetime.date,
        dt_end: datetime.date,
    ) -> pd.DataFrame:
        """Function to get data for all tickers in ticker_list from the specified db."""
        # Prepare variables
        df = pd.DataFrame()

        # Get data
        for ticker in ticker_list:
            try:
                df_ticker = pd.DataFrame(db[ticker].find(condition))
                df_ticker.loc[:, "ticker"] = ticker
                df = df.append(df_ticker)
            except Exception:
                print(ticker, "failed")
                continue

        # Set index
        df.set_index("_id", inplace=True)

        return df

    def _check_data(self, df: pd.DataFrame) -> None:
        """Function to check that the data meets all our requirements."""
        # Check that required columns exist
        if "date" not in df.columns:
            raise ValueError("df must be contain a 'date' column!")
        if "ticker" not in df.columns:
            raise ValueError("df must be contain a 'ticker' column!")

        # Check that index is _id
        if df.index.name != "_id":
            raise ValueError("df index must be '_id'!")

    def usa_alphavantage_eod(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from alphavantage based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ rd = RawData(mongodbkey)
            $ data = rd.usa_alphavantage_eod(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list, dt_start, dt_end = self._adjust_inputs(
            ticker_list, dt_start, dt_end
        )

        # Define function specific constants
        db_name = "USA-alphavantage-eod"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y-%m-%dT%X+00:00"),
                "$lte": dt_end.strftime("%Y-%m-%dT%X+00:00"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = self._get_data(db, condition, ticker_list, dt_start, dt_end)

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
            $ rd = RawData(mongodbkey)
            $ data = rd.usa_iex_1min(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list, dt_start, dt_end = self._adjust_inputs(
            ticker_list, dt_start, dt_end
        )

        # Define function specific constants
        db_name = "USA-IEX-1min"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y-%m-%dT%X+00:00"),
                "$lte": dt_end.strftime("%Y-%m-%dT%X+00:00"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = self._get_data(db, condition, ticker_list, dt_start, dt_end)

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
            $ rd = RawData(mongodbkey)
            $ data = rd.usa_yahoo_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list, dt_start, dt_end = self._adjust_inputs(
            ticker_list, dt_start, dt_end
        )

        # Define function specific constants
        db_name = "USA-yahoo-api"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y-%m-%dT%X+00:00"),
                "$lte": dt_end.strftime("%Y-%m-%dT%X+00:00"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = self._get_data(db, condition, ticker_list, dt_start, dt_end)

        # Check data
        self._check_data(df)

        return df

    def usa_finviz_api(
        self,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from finviz based on time and ticker selection.

        Example:
            $ mongodbkey = "" # your mongodbkey
            $ rd = RawData(mongodbkey)
            $ data = rd.usa_finviz_api(ticker_list = ["MSFT", "AAPL"])
        """
        # Check inputs
        self._check_inputs(ticker_list, dt_start, dt_end)

        # Adjust inputs
        ticker_list, dt_start, dt_end = self._adjust_inputs(
            ticker_list, dt_start, dt_end
        )

        # Define function specific constants
        db_name = "USA-finviz-api"
        time_key = "date"
        condition = {
            time_key: {
                "$gte": dt_start.strftime("%Y-%m-%dT%X+00:00"),
                "$lte": dt_end.strftime("%Y-%m-%dT%X+00:00"),
            }
        }

        # Fetch data
        db = self.client[db_name]
        df = self._get_data(db, condition, ticker_list, dt_start, dt_end)

        # Check data
        self._check_data(df)

        return df

    def bors_data(
        self,
        granularity: str,
        ticker_list: list,
        dt_start: datetime.datetime = None,
        dt_end: datetime.datetime = None,
    ) -> pd.DataFrame:
        """Fetches raw data from bors data source based on time and ticker selection."""
        iter_len = len(ticker_list)

        if granularity == "quarterly":
            collection = self.client["bors-data-quarterly"]
        if granularity == "yearly":
            collection = self.client["bors-data-yearly"]
        if granularity == "daily":
            collection = self.client["bors-data-daily"]

        if iter_len > 1:  # more than one tickers
            for x in range(0, iter_len):
                iter_query_result = pd.DataFrame(
                    collection[ticker_list[x]].find_one(
                        {"date": {"$gte": dt_start, "$lte": dt_end}}
                    )
                )

                iter_query_result.columns = [
                    str(col) + "_" + str(ticker_list[x])
                    for col in iter_query_result.columns
                ]

                if x == 0:
                    query_result = iter_query_result
                else:  # Joing in to dataframe
                    query_result = query_result.merge(iter_query_result, on="date")

        else:
            query_result = pd.DataFrame(
                collection[ticker_list[0]].find_one(
                    {"date": {"$gte": dt_start, "$lte": dt_end}}
                )
            )

        return query_result
