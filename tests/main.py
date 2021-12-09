import asyncio
import datetime
import json
import logging
import time

from tradingdmp.data.clf.price_perc_chg.app_data import DataAlpacaPocCat


async def main() -> None:

    # CONFIG
    config_path_private = ".config/config.private.json"
    config_path_public = ".config/config.dev.json"

    # Load mongodbkey and alpaca keys from private config
    with open(config_path_private) as f:
        data = json.load(f)
        mongodbkey = data["mongodbkey"]
        alpaca_env = data["alpaca_env"]

    # Load tickerlist from config
    with open(config_path_public) as f:
        data = json.load(f)
        n_ppc_per_row = data["n_ppc_per_row"]
        ticker_list = data["ticker_list"]

    def get_last_weekday() -> datetime.datetime:
        """Auxilary function to return last weekday relative to today."""
        today = datetime.datetime.today()
        days_offset = max(1, (today.weekday() + 6) % 7 - 3)
        timedelta = datetime.timedelta(days_offset)
        last_wd = today - timedelta
        return last_wd

    adata = DataAlpacaPocCat(mongodbkey)

    dt_end = get_last_weekday()  # yesterday
    dt_start = dt_end - datetime.timedelta(days=30)

    tic = time.perf_counter()
    data_dict = await adata.get_data(
        ticker_list,
        dt_start,
        dt_end,
        dt_end_required=True,
        return_last_date=True,
        return_nonlast_dates=False,
        return_training_dfs=False,
        return_date_col=False,
        return_ticker_col=False,
        n_ppc_per_row=n_ppc_per_row,
    )
    toc = time.perf_counter()

    logging.warning(f"Function ran in {toc - tic:0.4f} seconds.")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
