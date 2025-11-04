import numpy as np
import pytest
from tpil_calculate_tstat import calculate_tstat


def test_calculate_tstat_two_groups():
    # 2x2x2 image, 8 scans, 4 subjects, 2 groups (2 subjects per group)
    data_4d = np.zeros((2, 2, 2, 8))
    # Group A: s1, s2; Group B: s3, s4
    # s1: scans 0,1; s2: scans 2,3; s3: scans 4,5; s4: scans 6,7
    data_4d[..., 0] = 1  # s1, scan1
    data_4d[..., 1] = 1  # s1, scan2
    data_4d[..., 2] = 1  # s2, scan1
    data_4d[..., 3] = 1  # s2, scan2
    data_4d[..., 4] = 2  # s3, scan1
    data_4d[..., 5] = 2  # s3, scan2
    data_4d[..., 6] = 2  # s4, scan1
    data_4d[..., 7] = 2  # s4, scan2
    subjects = np.array(['s1', 's1', 's2', 's2', 's3', 's3', 's4', 's4'])
    groups = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    tstat_map, pval_map = calculate_tstat(data_4d, subjects, groups)
    # All voxels: group B mean = 2, group A mean = 1, but zero variance, so t = -inf
    assert np.all(np.isneginf(tstat_map))
    assert tstat_map.shape == (2, 2, 2)
    assert pval_map.shape == (2, 2, 2)


def test_calculate_tstat_invalid_groups():
    data_4d = np.zeros((2, 2, 2, 3))
    subjects = np.array(['s1', 's1', 's2'])
    groups = np.array(['A', 'A', 'A'])  # Only one group
    with pytest.raises(ValueError):
        calculate_tstat(data_4d, subjects, groups)


def test_calculate_tstat_nan_handling():
    # 2x2x2 image, 8 scans, 4 subjects, 2 groups (2 subjects per group)
    data_4d = np.ones((2, 2, 2, 8))
    data_4d[0, 0, 0, 0] = np.nan
    subjects = np.array(['s1', 's1', 's2', 's2', 's3', 's3', 's4', 's4'])
    groups = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    tstat_map, pval_map = calculate_tstat(data_4d, subjects, groups)
    # Only one voxel is nan, the rest should be -inf or nan (degenerate t-test)
    assert np.isnan(tstat_map[0, 0, 0])
    # Accept -inf or nan for degenerate t-test
    val = tstat_map[1, 1, 1]
    assert np.isnan(val) or np.isneginf(val)
