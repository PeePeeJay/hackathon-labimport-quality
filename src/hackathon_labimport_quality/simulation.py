# normal data


# categorical data

# ordinal data


import random
from collections.abc import Sequence


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
