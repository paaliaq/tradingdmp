"""Base class for all model classes."""

import numpy as np


def add(a: float, b: float) -> float:
    """Add two values a and b.

    Args:
      a: The argument which could be a number or could not be.
      b: The argument which could be a number or could not be.

    Raises:
      ValueError: If a or b < 0
    """
    if a < 0 or b < 0:
        raise ValueError

    return a + b


if __name__ == "__main__":

    print(np.random.randint(2, size=10))
    a = 2
    b = 5
    c = add(a, b)
    print(c)
