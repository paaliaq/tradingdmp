"""utility function used for DMP."""
from datetime import datetime
import pymongo
import pandas as pd


class RawData:
    """Class used for fetching raw data from database."""

    def __init__(self, mongodbkey: str):
        """Initializes connection to mongodb."""
        self.db = pymongo.MongoClient(mongodbkey)

    def bors_data(
        self,
        granularity: str,
        ticker_list: list,
        dt_start: datetime = datetime(2000, 1, 1),
        dt_end: datetime = datetime.today(),
    ) -> pd.DataFrame:
        """Fetches raw data from bors data source based on time and ticker selection."""
        iter_len = len(ticker_list)

        if granularity == "yearly":
            collection = self.db["bors-data-yearly"]
            time_key = "report_End_Date"
        if granularity == "quarterly":
            collection = self.db["bors-data-quarterly"]
            time_key = "report_End_Date"
        if granularity == "daily":
            collection = self.db["bors-data-daily"]
            time_key = "Time"

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

            iter_query_result.set_index(time_key, inplace=True)

            iter_query_result.columns = [
                str(col) + "_" + str(ticker_list[x])
                for col in iter_query_result.columns
            ]

            if x == 0:
                query_result = iter_query_result
            else:  # Join in to dataframe
                query_result = query_result.join(
                    iter_query_result, how="outer", on=time_key
                )

        return query_result
