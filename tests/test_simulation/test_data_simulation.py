import numpy as np

from hackathon_labimport_quality.simulation import valid_ordinal_data


def test_valid_ordinal_data():
    number = 1000
    levels = ["red", "blue", "green"]
    weights = [0.1, 0.4, 0.5]
    seed = 1337

    sample = valid_ordinal_data(number, levels, weights, seed)

    assert len(sample) == number
    assert all(value in levels for value in sample)


def test_valid_ordinal_data_frequencies_match_weights():
    number = 10_000
    levels = ["red", "blue", "green"]
    weights = [0.1, 0.4, 0.5]
    seed = 1337

    sample = valid_ordinal_data(number, levels, weights, seed)

    values, counts = np.unique(sample, return_counts=True)
    observed = dict(zip(values, counts / number))

    # np.unique sorts alphabetically, so align by level name, not position.
    # A level that was never sampled is missing from `observed` -> frequency 0.
    frequencies = np.array([observed.get(lvl, 0.0) for lvl in levels])

    # Std of a proportion is <= sqrt(0.25 / n) = 0.005 here; 0.02 is ~4 sigma.
    assert np.allclose(frequencies, weights, atol=0.02)
