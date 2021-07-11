"""Application classes for policies that are used in our trading apps."""

from typing import Any, Dict, List

import numpy as np
from tradingdmp.policy.clf.base_policy import BasePolicy

# import sys
# sys.path.append("../")
# from base_policy import BasePolicy


def get_arange_score(v: List[float]) -> float:
    """Method for returning a position."""
    n = len(v)
    rank_vector = np.arange(n) + 1
    v_vector = np.array(v)
    score = v_vector @ rank_vector
    return score


class ArangeScore(BasePolicy):
    """Simple policy for first prototype."""

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
                Example: {"ticker1:" [0.1, 0.25, 0.25, 0.25, 0.5],
                "ticker2": [0.1, 0.2, 0.2, 0.3, 0.2], ...,
                "tickerN": [0.1, 0.2, 0.3, 0.3, 0.1]}
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            output_dict: A dictionary with key-value pairs, where each key is a ticker
            symbol and each value is another dict. This other dict must contain the
            key "weight", representing the percentage of the depot value (cash + value
            of all investments) of the trader that is supposed to be allocated to the
            respective ticker symbol by the trader.
            Example: {"ticker1": 0.1, "ticker2": 0.05, ..., "tickerN": 0.0}

        """
        n = len(input_dict)

        # Add score and sort
        adj_dict = {}
        for k, v in input_dict.items():
            adj_dict[k] = {"cls_prob": input_dict[k], "score": get_arange_score(v)}
        adj_dict_sorted = sorted(
            adj_dict.items(), key=lambda x: -x[1]["score"]  # type: ignore
        )

        # Subset to the top 10
        adj_dict_sorted = adj_dict_sorted[0 : min(10, n)]

        # Get positions
        sum_of_scores = sum([x[1]["score"] for x in adj_dict_sorted])
        positions = {
            x[0]: x[1]["score"] / sum_of_scores for x in adj_dict_sorted  # type: ignore
        }
        return positions
