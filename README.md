# tpil_i2c2_filter

This repository contains scripts for filtering and calculating I2C2 clusters for TBSS analysis.

## Files
- `tpil_i2c2_filter.py`: Main filtering script.
- `calculate_i2c2_cluster.py`: Cluster calculation script.

## Usage

### Setup

1. Clone the repository:
	```sh
	git clone https://github.com/yourusername/tpil_i2c2_filter.git
	cd tpil_i2c2_filter
	```
2. (Optional) Create and activate a virtual environment:
	```sh
	python3 -m venv venv
	source venv/bin/activate
	```
3. Install dependencies:
	```sh
	pip install -r requirements.txt
	```

### Run Example

To run the main filtering script:

```sh
python tpil_i2c2_filter.py \
	--nifti_4d <path_to_4d_nifti.nii> \
	--visit_file <path_to_visit_ids.txt> \
	--subject_file <path_to_subject_ids.txt> \
	--group_file <path_to_group_file.txt> \
	--output_file <output_clusters.nii.gz> \
	[--stat_map <path_to_stat_map.nii>] \
	[--stat_threshold <float>] \
	[--size_threshold <int>] \
	[--i2c2_threshold <float>] \
	[--one_way] \
	[--mask <mask.nii.gz>]
```

#### Arguments

- `--nifti_4d` (required): Path to 4D NIfTI file containing scans in the 4th dimension.
- `--visit_file` (required): Text file with visit IDs (one per scan).
- `--subject_file` (required): Text file with subject IDs (one per scan).
- `--group_file` (required): Text file with group labels (one per scan, e.g. clbp/con).
- `--output_file` (required): Path to output NIfTI file for retained clusters.
- `--stat_map` (optional): Path to group difference/statistics map (NIfTI format). If not provided, the script will calculate the t-statistic map from the 4D NIfTI and group file.
- `--stat_threshold` (optional, default: 2.0): Threshold for statistics map; clusters below this value are ignored.
- `--size_threshold` (optional, default: 50): Minimum cluster size in voxels; smaller clusters are ignored.
- `--i2c2_threshold` (optional, default: 0.7): Minimum I2C2 value for clusters to be kept.
- `--one_way` (optional): Use one-way thresholding (stat >= threshold only). Default is two-way (stat >= threshold or stat <= -threshold).
- `--mask` (optional): NIfTI mask file. Only voxels within mask > 0 are analyzed.

Example:

```sh
python tpil_i2c2_filter.py \
	--nifti_4d scans_4d.nii.gz \
	--visit_file visits.txt \
	--subject_file subjects.txt \
	--group_file groups.txt \
	--output_file filtered_clusters.nii.gz \
	--stat_threshold 2.5 \
	--size_threshold 100 \
	--i2c2_threshold 0.8 \
	--mask mask.nii.gz
```

Use `python tpil_i2c2_filter.py --help` for more details on arguments and usage.

## License

See `LICENSE` for details.
