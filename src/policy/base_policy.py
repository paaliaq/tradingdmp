"""Base class for all policy classes."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BasePolicy(ABC):
    """Base class for policies that return a position."""

    @abstractmethod
    def get_position(self, y: pd.DataFrame, *args: Any, **kwargs: Any) -> int:
        """Method for returning a position.

        This function should fit the model given some training data x and y.

        Args:
            y (pd.DataFrame): Predicted targets in data frame of shape (1, d)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            qty (int): Quantity of shares of the particular ticker symbol that
                    should be owned by the trader from the next time step onwards.
                    If qty == 0, the trader should sell all shares if it owns any.
                    If qty > 0, the trader has to buy shares unless it already owns
                    qty shares of the particular ticker.

        """
        pass
