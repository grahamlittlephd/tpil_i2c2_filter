import numpy as np
import pytest
from tpil_calculate_i2c2 import compute_i2c2


def test_compute_i2c2_perfect_reliability():
    # 2 subjects, 2 visits, 3 voxels, all values identical per subject
    data = np.array([
        [[1, 2, 3], [1, 2, 3]],  # subject 1
        [[4, 5, 6], [4, 5, 6]],  # subject 2
    ])
    subjects = ['s1', 's2']
    visits = ['v1', 'v2']
    i2c2 = compute_i2c2(data, subjects, visits)
    assert np.isclose(i2c2, 1.0)


def test_compute_i2c2_no_reliability():
    # 2 subjects, 2 visits, 3 voxels, random noise (no reliability)
    np.random.seed(0)
    data = np.random.randn(2, 2, 3)
    subjects = ['s1', 's2']
    visits = ['v1', 'v2']
    # Should be low, but not exactly 0 due to randomness
    i2c2 = compute_i2c2(data, subjects, visits)
    assert 0 <= i2c2 <= 1


def test_compute_i2c2_nan_handling():
    # 2 subjects, 2 visits, 2 voxels, one NaN
    data = np.array([
        [[1, np.nan], [1, 2]],
        [[3, 4], [3, 4]],
    ])
    subjects = ['s1', 's2']
    visits = ['v1', 'v2']
    i2c2 = compute_i2c2(data, subjects, visits)
    assert not np.isnan(i2c2)


def test_compute_i2c2_zero_variance():
    # All values are the same, variance is zero, should return nan
    data = np.ones((2, 2, 2))
    subjects = ['s1', 's2']
    visits = ['v1', 'v2']
    i2c2 = compute_i2c2(data, subjects, visits)
    assert np.isnan(i2c2)
