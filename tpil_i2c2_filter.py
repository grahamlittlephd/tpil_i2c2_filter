import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage import label
from tpil_calculate_i2c2 import compute_i2c2


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

    # Load maps and info
    import nibabel as nib
    img_4d = nib.load(args.nifti_4d)
    data_4d = img_4d.get_fdata()
    subjects = np.loadtxt(args.subject_file, dtype=str)
    visits = np.loadtxt(args.visit_file, dtype=str)
    stat_img = None
    if args.stat_map:
        stat_img = nib.load(args.stat_map)
        stat_data = stat_img.get_fdata()
    else:
        groups = np.loadtxt(args.group_file, dtype=str)
        from tpil_calculate_tstat import calculate_tstat
        stat_data, _ = calculate_tstat(data_4d, subjects, groups)
        stat_img = nib.Nifti1Image(stat_data, img_4d.affine, img_4d.header)
        print("Calculated t-statistic map from 4D NIfTI and group file.")

    # Apply mask if provided
    mask = None
    if args.mask:
        mask_img = nib.load(args.mask)
        mask = mask_img.get_fdata() > 0
        print(f"Loaded mask from {args.mask}")
        stat_data = np.where(mask, stat_data, 0)

    # Threshold statistics map
    if args.one_way:
        stat_mask = stat_data >= args.stat_threshold
        print(f"Using one-way threshold: stat >= {args.stat_threshold}")
    else:
        stat_mask = (stat_data >= args.stat_threshold) | (
            stat_data <= -args.stat_threshold)
        print(
            f"Using two-way threshold: stat >= {args.stat_threshold} or stat <= {-args.stat_threshold}")
    labeled, n_clusters = label(stat_mask)
    print(
        f"Found {n_clusters} clusters passing statistics threshold prior to filtering.")

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
        avg_stat = np.mean(stat_data[cluster_mask])
        results[-1]['avg_stat'] = avg_stat
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
    import csv
    csv_path = args.output_file
    if csv_path.endswith('.nii') or csv_path.endswith('.nii.gz'):
        csv_path = csv_path.replace('.nii.gz', '.csv').replace('.nii', '.csv')
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
                                "cluster_index", "size", "i2c2", "avg_stat"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Cluster info written to {csv_path}")


if __name__ == "__main__":
    main()
