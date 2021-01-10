"""utility function used for DMP."""
import datetime
from typing import List
import pymongo
import pandas as pd


class RawData:
    """Class used for fetching raw data from database."""

    def __init__(self, mongodb_key: str):
        """Initializes connection to mongodb."""
        # type checks
        if not isinstance(mongodb_key, str):
            print("mongodb_key must be of type string!")

        self.db = pymongo.MongoClient(mongodb_key)

    def bors_data(
        self,
        granularity: str,
        ticker_list: List[str],
        dt_start: datetime.date = datetime.datetime(2000, 1, 1),
        dt_end: datetime.date = datetime.datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from bors data source based on time and ticker selection."""
        iter_len = len(ticker_list)

        # type checks
        if not isinstance(granularity, str):
            raise TypeError("granularity must be of type string!")

        if not isinstance(dt_start, datetime.datetime):
            raise TypeError("dt_start must be of type datetime!")

        if not isinstance(dt_end, datetime.datetime):
            raise TypeError("dt_end must be of type datetime!")

        if not isinstance(ticker_list, list):
            raise TypeError("ticker_list must be of type list!")

        # Logical check start end
        if dt_start > dt_end:
            raise ValueError("dt_start must be less than dt_end!")

        # Remove duplicates from list
        ticker_set = set(ticker_list)
        ticker_list = list(ticker_set)

        if granularity not in ["yearly", "quarterly", "daily"]:
            raise ValueError(
                "granularity must any of: 'daily', 'quarterly' or 'yearly'"
            )

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

        collection = self.db[granularity_dict[granularity]["collection_name"]]
        time_key = granularity_dict[granularity]["time_key"]

        iter_len = len(ticker_list)
        for x in range(0, iter_len):
            try:
                iter_query_result = pd.DataFrame(
                    collection[ticker_list[x]].find(
                        {time_key: {"$gte": dt_start, "$lte": dt_end}}
                    )
                )
            except Exception:
                print(ticker_list[x], "failed")
                continue

            iter_query_result.rename({time_key: "date"}, axis=1, inplace=True)
            iter_query_result.set_index("_id", inplace=True)
            iter_query_result["ticker"] = ticker_list[x]

            if x == 0:
                query_result = iter_query_result
            else:  # Join in to dataframe
                query_result = pd.concat([iter_query_result, iter_query_result])

        return query_result
