"""Base classes for policies."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePolicy(ABC):
    """Base class for policies."""

    @abstractmethod
    def get_position(
        self, input_dict: Dict[str, Dict[str, Any]], *args: Any, **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Method for returning a position.

        This function should take a dict that contains, among others, the predicted
        price percentage change perc_chg for each ticker symbol that is to be considered
        for trading. Then, this function should return a dict that contains, among
        others, the percentage of the depot (cash + value) that is to be allocated to
        the respective ticker symbols by the trader.

        Args:
            input_dict: A dictionary with key-value pairs, where each key is a ticker
                symbol and each value is another dict. This other dict must contain the
                key "perc_chg" representing the predicted price percentage change.
                Example: {"ticker1:" {"perc_chg": 0.01}, "ticker2": {"perc_chg": 0.4},
                ..., "tickerN": {"perc_chg": -0.25}}
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            output_dict: A dictionary with key-value pairs, where each key is a ticker
            symbol and each value is another dict. This other dict must contain the
            key "weight", representing the percentage of the depot value (cash + value
            of all investments) of the trader that is supposed to be allocated to the
            respective ticker symbol by the trader. One of the keys must be called
            "cash" and holds the percentage of depot value to be allocated to cash.
            Example: {"cash: {"weight": 0.125}, "ticker1:" {"weight": 0.1},
            "ticker2": {"weight": 0.05}, ..., "tickerN": {"weight": 0.0}}

        """
        pass
