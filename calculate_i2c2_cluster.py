import numpy as np
import nibabel as nib
import pandas as pd
import argparse


def compute_i2c2(data, subjects, visits):
    """
    Compute the image intraclass correlation coefficient (I2C2) as in Shou et al. (2013).
    data: array of shape (n_subjects, n_visits, n_voxels)
    """
    n_subjects, n_visits, n_voxels = data.shape

    # Subject means (n_subjects x n_voxels)
    subj_means = np.nanmean(data, axis=1)

    # Grand mean (1 x n_voxels)
    grand_mean = np.nanmean(subj_means, axis=0)

    # Between-subject sum of squares (variance of subject means)
    ss_between = np.nansum((subj_means - grand_mean) ** 2)

    # Within-subject sum of squares (variance of residuals)
    ss_within = 0
    for i in range(n_subjects):
        residuals = data[i, :, :] - subj_means[i][None, :]
        ss_within += np.nansum(residuals ** 2)

    # I2C2
    denom = ss_between + ss_within
    i2c2 = ss_between / denom if denom > 0 else np.nan
    return i2c2


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average image ICC (I2C2) within an ROI.")
    parser.add_argument("--nifti_file", required=True,
                        help="Path to 4D NIfTI file (scans in 4th dimension)")
    parser.add_argument("--roi_mask", required=True,
                        help="Path to ROI mask NIfTI file")
    parser.add_argument("--subject_file", required=True,
                        help="Text file with subject IDs (one per scan)")
    parser.add_argument("--visit_file", required=True,
                        help="Text file with visit IDs (one per scan)")
    parser.add_argument("--group_file", required=False,
                        help="Text file with group labels (optional)")
    args = parser.parse_args()

    # Load data
    img = nib.load(args.nifti_file)
    data = img.get_fdata()  # shape: (X, Y, Z, N)
    mask = nib.load(args.roi_mask).get_fdata().astype(bool)
    subjects = np.loadtxt(args.subject_file, dtype=str)
    visits = np.loadtxt(args.visit_file, dtype=str)

    # Check dimensions
    n_scans = data.shape[3]
    if not (len(subjects) == len(visits) == n_scans):
        raise ValueError(
            "Mismatch between number of scans and subject/visit files.")

    # Get unique subjects and visits
    unique_subjects = np.unique(subjects)
    unique_visits = np.unique(visits)

    # Find subjects with all visits
    subjects_with_all_visits = [subj for subj in unique_subjects if np.sum((subjects == subj)) == len(unique_visits)
                                and all(np.sum((subjects == subj) & (visits == v)) == 1 for v in unique_visits)]

    if len(subjects_with_all_visits) == 0:
        raise ValueError("No subjects have all visits/scans.")

    # Arrange data as (n_subjects, n_visits, n_voxels)
    n_subjects = len(subjects_with_all_visits)
    n_visits = len(unique_visits)
    n_voxels = mask.sum()
    data_roi = data[mask]  # shape: (n_voxels, n_scans)

    # Build subject x visit x voxel array
    data_matrix = np.full((n_subjects, n_visits, n_voxels), np.nan)
    for i, subj in enumerate(subjects_with_all_visits):
        for j, visit in enumerate(unique_visits):
            idx = np.where((subjects == subj) & (visits == visit))[0]
            if len(idx) == 1:
                data_matrix[i, j, :] = data_roi[:, idx[0]]
            else:
                continue  # Should not happen due to filtering

    # Compute I2C2
    i2c2 = compute_i2c2(data_matrix, subjects_with_all_visits, unique_visits)
    print(f"I2C2 within ROI: {i2c2:.4f}")


if __name__ == "__main__":
    main()
