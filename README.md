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
	--stat_map <path_to_stat_map.nii> \
	--nifti_4d <path_to_4d_nifti.nii> \
	--visit_file <path_to_visit_ids.txt> \
	--subject_file <path_to_subject_ids.txt> \
	[--stat_threshold <float>] \
	[--size_threshold <int>] \
	[--i2c2_threshold <float>]
```

#### Arguments

- `--stat_map` (required): Path to group difference/statistics map (NIfTI format).
- `--nifti_4d` (required): Path to 4D NIfTI file containing scans in the 4th dimension.
- `--visit_file` (required): Text file with visit IDs (one per scan).
- `--subject_file` (required): Text file with subject IDs (one per scan).
- `--stat_threshold` (optional, default: 2.0): Threshold for statistics map; clusters below this value are ignored.
- `--size_threshold` (optional, default: 50): Minimum cluster size in voxels; smaller clusters are ignored.
- `--i2c2_threshold` (optional, default: 0.7): Minimum I2C2 value for clusters to be kept.

Example:

```sh
python tpil_i2c2_filter.py \
	--stat_map stats.nii.gz \
	--nifti_4d scans_4d.nii.gz \
	--visit_file visits.txt \
	--subject_file subjects.txt \
	--stat_threshold 2.5 \
	--size_threshold 100 \
	--i2c2_threshold 0.8
```

Use `python tpil_i2c2_filter.py --help` for more details on arguments and usage.

## License

See `LICENSE` for details.
