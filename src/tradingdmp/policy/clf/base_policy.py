"""Base classes for policies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePolicy(ABC):
    """Base class for policies."""

    @abstractmethod
    def get_position(
        self, input_dict: Dict[str, List[float]], *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        """Method for returning a position.

        This function should take a dict that contains, among others, the predicted
        class probabilities for each ticker symbol that is to be considered
        for trading. Then, this function should return a dict that contains
        the percentage of the depot (cash + value) that is to be allocated to
        the respective ticker symbols by the trader.

        Args:
            input_dict: A dictionary with key-value pairs, where each key is a ticker
                symbol and each value is another dict. This other dict must contain the
                key "perc_chg" representing the predicted price percentage change.
                Example: {"ticker1:" [0.01, 0.25, 0.25, 0.25, 0.1, 0.05],
                "ticker2": [0.01, 0.2, 0.2, 0.3, 0.15, 0.05], ...,
                "tickerN": [0.01, 0.2, 0.3, 0.3, 0.05, 0.05]}
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            output_dict: A dictionary with key-value pairs, where each key is a ticker
            symbol and each value is another dict. This other dict must contain the
            key "weight", representing the percentage of the depot value (cash + value
            of all investments) of the trader that is supposed to be allocated to the
            respective ticker symbol by the trader.
            Example: {ticker1": 0.1, "ticker2": 0.05, ..., "tickerN": 0.0}

        """
        pass
