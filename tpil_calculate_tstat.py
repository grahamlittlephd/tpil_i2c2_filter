import numpy as np
import nibabel as nib
import argparse
from scipy.stats import ttest_ind
import os


def assert_file_exists(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} file not found: {path}")


def assert_nifti_file(path, desc):
    assert_file_exists(path, desc)
    if not (path.endswith('.nii') or path.endswith('.nii.gz')):
        raise ValueError(
            f"{desc} must be a NIfTI file (.nii or .nii.gz): {path}")


def calculate_tstat(data_4d, subjects, groups):
    """
    Calculate t-statistic and p-value maps for group difference.
    data_4d: numpy array (X, Y, Z, n_scans)
    subjects: array of subject IDs (n_scans)
    groups: array of group labels (n_scans)
    Returns: tstat_map, pval_map
    """
    unique_subjects = np.unique(subjects)
    subject_averages = []
    subject_group_labels = []
    for subj in unique_subjects:
        idx = np.where(subjects == subj)[0]
        subj_data = data_4d[..., idx]
        avg_data = np.mean(subj_data, axis=-1)
        subject_averages.append(avg_data)
        subject_group_labels.append(groups[idx[0]])
    subject_averages = np.stack(subject_averages, axis=-1)
    subject_group_labels = np.array(subject_group_labels)
    unique_groups = np.unique(subject_group_labels)
    if len(unique_groups) != 2:
        raise ValueError(
            f"Expected exactly 2 groups, found: {unique_groups}")
    groupA, groupB = unique_groups
    groupA_idx = np.where(subject_group_labels == groupA)[0]
    groupB_idx = np.where(subject_group_labels == groupB)[0]
    groupA_data = subject_averages[..., groupA_idx]
    groupB_data = subject_averages[..., groupB_idx]
    shape = subject_averages.shape[:-1]
    groupA_flat = groupA_data.reshape(-1, groupA_data.shape[-1]).T
    groupB_flat = groupB_data.reshape(-1, groupB_data.shape[-1]).T
    tvals, pvals = ttest_ind(
        groupA_flat, groupB_flat, axis=0, nan_policy='omit')
    tstat_map = tvals.reshape(shape)
    pval_map = pvals.reshape(shape)

    return tstat_map, pval_map


def main():
    parser = argparse.ArgumentParser(
        description="Generate group difference statistics map from 4D NIfTI data. Averaging across visits prior to t-test.")
    parser.add_argument("--nifti_4d", required=True,
                        help="Path to 4D NIfTI file (scans in 4th dimension)")
    parser.add_argument("--subject_file", required=True,
                        help="Text file with subject IDs (one per scan)")
    parser.add_argument("--group_file", required=True,
                        help="Text file with group labels (one per scan, e.g. clbp/con)")
    parser.add_argument("--output_tstat", required=True,
                        help="Output NIfTI file for t-statistic map")
    parser.add_argument("--output_pval", required=True,
                        help="Output NIfTI file for p-value map")
    parser.add_argument("--mask", required=False,
                        help="Optional NIfTI mask file. Only voxels within mask > 0 are analyzed.")
    args = parser.parse_args()

    # Validate input files
    assert_nifti_file(args.nifti_4d, "4D NIfTI input")
    assert_file_exists(args.subject_file, "Subject list")
    assert_file_exists(args.group_file, "Group list")
    if args.mask:
        assert_nifti_file(args.mask, "Mask")
    if not (args.output_tstat.endswith('.nii') or args.output_tstat.endswith('.nii.gz')):
        raise ValueError(
            f"Output tstat file must be a NIfTI file (.nii or .nii.gz): {args.output_tstat}")
    if not (args.output_pval.endswith('.nii') or args.output_pval.endswith('.nii.gz')):
        raise ValueError(
            f"Output pval file must be a NIfTI file (.nii or .nii.gz): {args.output_pval}")

    img_4d = nib.load(args.nifti_4d)
    data_4d = img_4d.get_fdata()
    subjects = np.loadtxt(args.subject_file, dtype=str)
    groups = np.loadtxt(args.group_file, dtype=str)
    mask = None
    if args.mask:
        mask_img = nib.load(args.mask)
        mask = mask_img.get_fdata() > 0
        print(f"Loaded mask from {args.mask}")

    tstat_map, pval_map = calculate_tstat(data_4d, subjects, groups)

    # Apply mask if provided
    if mask is not None:
        tstat_map = np.where(mask, tstat_map, 0)
        pval_map = np.where(mask, pval_map, 1)
    tstat_img = nib.Nifti1Image(tstat_map, img_4d.affine, img_4d.header)
    pval_img = nib.Nifti1Image(pval_map, img_4d.affine, img_4d.header)
    nib.save(tstat_img, args.output_tstat)
    nib.save(pval_img, args.output_pval)
    print(f"Saved t-statistic map to {args.output_tstat}")
    print(f"Saved p-value map to {args.output_pval}")


if __name__ == "__main__":
    main()
