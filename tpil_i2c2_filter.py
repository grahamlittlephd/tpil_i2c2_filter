import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage import label
from calculate_i2c2_cluster import compute_i2c2


def main():
    parser = argparse.ArgumentParser(
        description="Threshold and filter clusters by size and I2C2.")
    parser.add_argument("--stat_map", required=True,
                        help="Path to group difference/statistics map (NIfTI)")
    parser.add_argument("--nifti_4d", required=True,
                        help="Path to 4D NIfTI file (scans in 4th dimension)")
    parser.add_argument("--visit_file", required=True,
                        help="Text file with visit IDs (one per scan)")
    parser.add_argument("--subject_file", required=True,
                        help="Text file with subject IDs (one per scan)")
    parser.add_argument("--stat_threshold", type=float, default=2.0,
                        help="Threshold for statistics map (default: 2.0)")
    parser.add_argument("--size_threshold", type=int, default=50,
                        help="Minimum cluster size (default: 50 voxels)")
    parser.add_argument("--i2c2_threshold", type=float, default=0.7,
                        help="Minimum I2C2 value for cluster (default: 0.7)")
    args = parser.parse_args()

    # Load maps and info
    stat_img = nib.load(args.stat_map)
    stat_data = stat_img.get_fdata()
    img_4d = nib.load(args.nifti_4d)
    data_4d = img_4d.get_fdata()
    subjects = np.loadtxt(args.subject_file, dtype=str)
    visits = np.loadtxt(args.visit_file, dtype=str)

    # Threshold statistics map
    mask = stat_data >= args.stat_threshold
    labeled, n_clusters = label(mask)
    print(f"Found {n_clusters} clusters above threshold {args.stat_threshold}")

    # Prepare output
    results = []
    for cluster_idx in range(1, n_clusters+1):
        cluster_mask = labeled == cluster_idx
        cluster_size = cluster_mask.sum()
        if cluster_size < args.size_threshold:
            continue
        # Extract data for cluster
        data_roi = data_4d[cluster_mask]  # shape: (n_voxels, n_scans)
        n_voxels = data_roi.shape[0]
        n_scans = data_roi.shape[1]
        # Get unique subjects and visits
        unique_subjects = np.unique(subjects)
        unique_visits = np.unique(visits)
        # Find subjects with all visits
        subjects_with_all_visits = [subj for subj in unique_subjects if np.sum((subjects == subj)) == len(unique_visits)
                                    and all(np.sum((subjects == subj) & (visits == v)) == 1 for v in unique_visits)]
        n_subjects = len(subjects_with_all_visits)
        n_visits = len(unique_visits)
        # Build subject x visit x voxel array
        data_matrix = np.full((n_subjects, n_visits, n_voxels), np.nan)
        for i, subj in enumerate(subjects_with_all_visits):
            for j, visit in enumerate(unique_visits):
                idx = np.where((subjects == subj) & (visits == visit))[0]
                if len(idx) == 1:
                    data_matrix[i, j, :] = data_roi[:, idx[0]]
        # Compute I2C2
        i2c2 = compute_i2c2(
            data_matrix, subjects_with_all_visits, unique_visits)
        if i2c2 < args.i2c2_threshold:
            continue
        results.append({
            'cluster_index': cluster_idx,
            'size': cluster_size,
            'i2c2': i2c2
        })
        print(f"Cluster {cluster_idx}: size={cluster_size}, I2C2={i2c2:.4f}")

    print(f"Kept {len(results)} clusters passing all thresholds.")
    # Optionally, save cluster masks or results here


if __name__ == "__main__":
    main()
