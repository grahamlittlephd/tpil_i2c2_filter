import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage import label
from tpil_calculate_i2c2 import compute_i2c2
import os
import csv


def assert_file_exists(path, desc):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} file not found: {path}")


def assert_nifti_file(path, desc):
    assert_file_exists(path, desc)
    if not (path.endswith('.nii') or path.endswith('.nii.gz')):
        raise ValueError(
            f"{desc} must be a NIfTI file (.nii or .nii.gz): {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Threshold and filter clusters by size and I2C2.")
    parser.add_argument("--nifti_4d", required=True,
                        help="Path to 4D NIfTI file (scans in 4th dimension)")
    parser.add_argument("--visit_file", required=True,
                        help="Text file with visit IDs (one per scan)")
    parser.add_argument("--subject_file", required=True,
                        help="Text file with subject IDs (one per scan)")
    parser.add_argument("--group_file", required=True,
                        help="Text file with group labels (one per scan, e.g. clbp/con)")
    parser.add_argument("--output_file", required=True,
                        help="Path to output NIfTI file for retained clusters")
    parser.add_argument("--stat_threshold", type=float, default=2.0,
                        help="Threshold for statistics map (default: 2.0)")
    parser.add_argument("--size_threshold", type=int, default=50,
                        help="Minimum cluster size (default: 50 voxels)")
    parser.add_argument("--i2c2_threshold", type=float, default=0.7,
                        help="Minimum I2C2 value for cluster (default: 0.7)")
    parser.add_argument("--one_way", action="store_true",
                        help="Use one-way thresholding (stat >= threshold only). Default is two-way (stat >= threshold or stat <= -threshold).")
    parser.add_argument("--stat_map", required=False,
                        help="Path to group difference/statistics map (NIfTI). If not provided, will be calculated from 4D NIfTI and group file.")
    parser.add_argument("--mask", required=False,
                        help="Optional NIfTI mask file. Only voxels within mask > 0 are analyzed.")
    args = parser.parse_args()

    # Validate input files
    assert_nifti_file(args.nifti_4d, "4D NIfTI input")
    assert_file_exists(args.visit_file, "Visit list")
    assert_file_exists(args.subject_file, "Subject list")
    assert_file_exists(args.group_file, "Group list")
    if args.stat_map:
        assert_nifti_file(args.stat_map, "Stat map")
    if args.mask:
        assert_nifti_file(args.mask, "Mask")
    if not (args.output_file.endswith('.nii') or args.output_file.endswith('.nii.gz')):
        raise ValueError(
            f"Output file must be a NIfTI file (.nii or .nii.gz): {args.output_file}")

    # Load subject, visit, group info
    with open(args.visit_file) as f:
        visits = np.array([line.strip() for line in f if line.strip()])
    with open(args.subject_file) as f:
        subjects = np.array([line.strip() for line in f if line.strip()])
    with open(args.group_file) as f:
        groups = np.array([line.strip() for line in f if line.strip()])
    if not (len(visits) == len(subjects) == len(groups)):
        raise ValueError(
            "Visit, subject, and group files must have the same number of lines.")

    # Load 4D data
    img_4d = nib.load(args.nifti_4d)
    data_4d = img_4d.get_fdata()
    n_scans = data_4d.shape[3]
    if n_scans != len(subjects):
        raise ValueError(
            f"4D NIfTI has {n_scans} volumes, but {len(subjects)} subjects listed.")

    # Apply mask if provided
    if args.mask:
        mask_img = nib.load(args.mask)
        mask = mask_img.get_fdata() > 0
    else:
        mask = np.ones(data_4d.shape[:3], dtype=bool)

    if args.stat_map:
        stat_img = nib.load(args.stat_map)
        stat_data = stat_img.get_fdata()
    else:
        # Compute mean difference between groups (simple t-like stat)
        group_labels = np.unique(groups)
        if len(group_labels) != 2:
            raise ValueError(
                "Exactly two groups required to compute stat map.")
        group1_idx = np.where(groups == group_labels[0])[0]
        group2_idx = np.where(groups == group_labels[1])[0]
        mean1 = np.nanmean(data_4d[..., group1_idx], axis=3)
        mean2 = np.nanmean(data_4d[..., group2_idx], axis=3)
        stat_data = mean1 - mean2
        stat_img = nib.Nifti1Image(stat_data, img_4d.affine, img_4d.header)

    # Threshold stat map
    if args.one_way:
        threshold_mask = (stat_data >= args.stat_threshold)
    else:
        threshold_mask = (stat_data >= args.stat_threshold) | (
            stat_data <= -args.stat_threshold)
    threshold_mask = threshold_mask & mask

    # Label clusters
    labeled, n_clusters = label(threshold_mask)
    results = []

    for cluster_idx in range(1, n_clusters + 1):
        cluster_mask = (labeled == cluster_idx)
        cluster_size = np.sum(cluster_mask)
        if cluster_size < args.size_threshold:
            continue

        # Extract data for this cluster across all scans
        cluster_voxels = np.where(cluster_mask)
        n_voxels = len(cluster_voxels[0])
        if n_voxels == 0:
            continue
        data_roi = data_4d[cluster_voxels[0],
                           cluster_voxels[1], cluster_voxels[2], :]
        data_roi = data_roi.T  # shape: n_scans x n_voxels

        # Find unique subjects and visits
        unique_subjects = np.unique(subjects)
        unique_visits = np.unique(visits)
        # Build subject x visit x voxel array
        subj_visit_matrix = np.full(
            (len(unique_subjects), len(unique_visits), n_voxels), np.nan)
        for i, subj in enumerate(unique_subjects):
            for j, visit in enumerate(unique_visits):
                idx = np.where((subjects == subj) & (visits == visit))[0]
                if len(idx) == 1:
                    subj_visit_matrix[i, j, :] = data_roi[idx[0], :]

        # Compute I2C2
        i2c2 = compute_i2c2(subj_visit_matrix, unique_subjects, unique_visits)
        if i2c2 < args.i2c2_threshold:
            continue

        avg_stat = np.mean(stat_data[cluster_mask])
        results.append({
            'cluster_index': cluster_idx,
            'size': cluster_size,
            'i2c2': i2c2,
            'avg_stat': avg_stat
        })
        print(
            f"Cluster {cluster_idx}: size={cluster_size}, I2C2={i2c2:.4f}, avg_stat={avg_stat:.4f} - kept")

    print(f"Kept {len(results)} clusters passing all thresholds.")
    # Create output map: clusters that pass filtering, voxel values = label ID
    output_map = np.zeros(stat_data.shape, dtype=np.int32)
    for r in results:
        output_map[labeled == r['cluster_index']] = r['cluster_index']
    out_img = nib.Nifti1Image(output_map, stat_img.affine, stat_img.header)
    nib.save(out_img, args.output_file)
    print(f"Output map written to {args.output_file}")

    # Also write cluster info to CSV (same path with .csv extension)
    csv_path = args.output_file
    if csv_path.endswith('.nii.gz'):
        csv_path = csv_path[:-7] + '.csv'
    elif csv_path.endswith('.nii'):
        csv_path = csv_path[:-4] + '.csv'
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
                                "cluster_index", "size", "i2c2", "avg_stat"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Cluster info written to {csv_path}")


if __name__ == "__main__":
    main()
