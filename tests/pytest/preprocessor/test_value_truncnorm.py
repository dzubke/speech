# third-party libraries
import numpy as np
import pytest
# project libraries
from speech.utils.signal_augment import get_value_from_truncnorm


# each of the functions below checks the values for the range of parameters used

def test_noise_inject():
    center = 0.0
    value_range = [0, 0.5]
    bounds = [0, 1.0]

    check_truncnorm(center, value_range, bounds)


def test_tempo_pertub():
    center = 1.0
    value_range = [0.85, 1.15]
    bounds = [0.0, 3.0]
    
    check_truncnorm(center, value_range, bounds)


def test_gain_pertub():
    center = 0.0
    value_range = [-6, 6]
    bounds = [-12, 12]
    
    check_truncnorm(center, value_range, bounds)


def test_pitch_pertub():
    center = 0.0
    value_range = [-400, 400]
    bounds = [-1200, 1200]
    
    check_truncnorm(center, value_range, bounds)


def test_gaussian_noise():
    center = 30
    value_range = [10, 50]
    bounds = [-70, 100]
    
    check_truncnorm(center, value_range, bounds)


def check_truncnorm(center, value_range, bounds):

    iterations = 100000
    test_values = list()

    bounds.sort()
    value_range.sort()

    range_diff = abs(value_range[0] - value_range[1])
    norm_std = range_diff/2

    for i in range(iterations):
        test_values.append(get_value_from_truncnorm(center, value_range, bounds))
    

    assert min(test_values) > bounds[0], "lower bound violated"
    assert max(test_values) < bounds[1], "upper bound violated"
    # TODO The bounds on the approx statements are so loose, they don't mean much. 
    # the main thing I wanted to test was the min and max values. testing for mean and std of truncated distributions is tricky.
    assert np.mean(test_values) == pytest.approx(center, abs=range_diff*1e-0), f"mean value {np.mean(test_values)} outside range"
    assert np.std(test_values) == pytest.approx(norm_std,  rel=0.5, abs=1e-1), f"std value {np.std(test_values)} outside range"