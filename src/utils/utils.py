"""utility function used for DMP."""
import datetime
import pymongo
import pandas as pd


class Rawdata:
    """Class used for fetching raw data from database."""

    def __init__(self, mongodbkey: str):
        """Initializes connection to mongodb."""
        self.db = pymongo.MongoClient(mongodbkey)

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
            collection = self.db["bors-data-quarterly"]
        if granularity == "yearly":
            collection = self.db["bors-data-yearly"]
        if granularity == "daily":
            collection = self.db["bors-data-daily"]

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
                else:  # Join in to dataframe
                    query_result = query_result.merge(iter_query_result, on="date")

        else:
            query_result = pd.DataFrame(
                collection[ticker_list[0]].find_one(
                    {"date": {"$gte": dt_start, "$lte": dt_end}}
                )
            )

        return query_result
