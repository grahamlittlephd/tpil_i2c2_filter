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
python tpil_i2c2_filter.py --input <input_file> --output <output_file> [other options]
```

To run the cluster calculation script:
```sh
python calculate_i2c2_cluster.py --input <input_file> --clusters <clusters_file> [other options]
```

See script docstrings or use `--help` for more details on arguments.

## License

See `LICENSE` for details.
