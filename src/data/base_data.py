"""Base class for all data classes."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Tuple

import pandas as pd


class BaseData(ABC):
    """Base class for data fetching and processing."""

    @abstractmethod
    def get_data(
        self,
        dt_start: Optional[datetime] = None,
        dt_end: Optional[datetime] = None,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Method for getting data that can be passed to a model or a policy.

        This function should fetch raw data, clean this data, conduct feature
        engineering and return features x and y as data frames in a tuple (x, y)

        Args:
            dt_start (datetime): All data after incl. dt_start is fetched.
                By default, dt_start is None, which means that data is fetched
                from the first available datetime onward.
            dt_end (datetime): All data until incl. dt_end is fetched.
                By default, dt_end is None, which means that data is fetched
                until the last the available datetime.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            (x, y): tuple with two pandas data frames: x of shape (n, m) and
                    y of shape (n, d)

        """
        pass
