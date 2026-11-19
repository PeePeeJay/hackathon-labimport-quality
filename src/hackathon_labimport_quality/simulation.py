import random
from collections.abc import Sequence

import numpy as np

def valid_normal_data(
    number: int,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate `number` values from a normal distribution N(mean, std**2).

    Pass `seed` for reproducible output.
    """
    if number < 0:
        raise ValueError("number must be non-negative")
    if std < 0:
        raise ValueError("std must be non-negative")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=number)


# categorical data
def valid_categorical_data(
    number: int,
    categories: Sequence[str] = ("A", "B", "C"),
    weights: Sequence[float] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Simulate `number` valid nominal values drawn from `categories`.

    Categories have no inherent order. `weights` gives the relative
    probability of each category (default: uniform).
    Pass `seed` for reproducible output.
    """
    if number < 0:
        raise ValueError("number must be non-negative")
    if not categories:
        raise ValueError("categories must not be empty")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must be unique")
    if weights is not None and len(weights) != len(categories):
        raise ValueError("weights must have the same length as categories")

    rng = random.Random(seed)
    return rng.choices(list(categories), weights=weights, k=number)


# ordinal data
def valid_ordinal_data(
    number: int,
    levels: Sequence[str] = ("negative", "trace", "+", "++", "+++"),
    weights: Sequence[float] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Simulate `number` valid ordinal values drawn from the ordered `levels`.

    `weights` gives the relative probability of each level (default: uniform).
    Pass `seed` for reproducible output.
    """
    if number < 0:
        raise ValueError("number must be non-negative")
    if not levels:
        raise ValueError("levels must not be empty")
    if weights is not None and len(weights) != len(levels):
        raise ValueError("weights must have the same length as levels")

    rng = random.Random(seed)
    return rng.choices(list(levels), weights=weights, k=number)

