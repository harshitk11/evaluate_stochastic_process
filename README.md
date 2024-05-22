# Synthetic Forest Fire and Real World Wildfire Experiments

This repository contains the code for experiments and the data generation process for both synthetic forest fire datasets and real world wildfire datasets.

## Repository Components

The repository is divided into two main components:

1. **Synthetic Forest Fire Evolution**: Detailed information and documentation can be found in the [README](synthetic_dataset_experiments/README.md).
2. **Real World Wildfire Data**: Detailed information and documentation can be found in the [README](real_wildfire_dataset_experiments/README.md).

### Synthetic Forest Fire Data Generation

The synthetic forest fire data generation process is thoroughly documented. For more information, refer to the [README](synthetic_dataset_experiments/NetLogo-Synthetic-Dataset-Generator/README.md) in the `NetLogo-Synthetic-Dataset-Generator` directory.

## Code for Generating Results

Below are the links to the code used for generating the results presented in various figures and tables:

### Synthetic Forest Fire Data

- **Figure 4**: [Code](synthetic_dataset_experiments/NetLogo-Synthetic-Dataset-Generator/pyNetLogo_script.py)
- **Figures 5, 13**: [Code](synthetic_dataset_experiments/experiments/stochastic_process_matrix_eval.py)
- **Figures 6, 14**: [Code](synthetic_dataset_experiments/experiments/ece_counterexample_study.py)
- **Figures 10, 11**: [Code](synthetic_dataset_experiments/experiments/reliability_analysis.py)
- **Figure 12**: [Code](synthetic_dataset_experiments/experiments/predictive_difficulty_score_analysis.py)
- **Figure 15**: [Code](synthetic_dataset_experiments/experiments/metric_evolution_vs_num_samples.py)
- **Figure 6 (inset), 17**: [Code](synthetic_dataset_experiments/experiments/calibration_curve_instant.py)

### Real World Wildfire Data

- **Figure 16**: [Code](real_wildfire_dataset_experiments/experiments/calibration_curve_baselines_compare.py)
- **Tables 1, 2**: [Code](real_wildfire_dataset_experiments/experiments/ece_score_table_generator.py)

## Directory Structure

- **synthetic_dataset_experiments/**: Contains the synthetic forest fire data generation and analysis scripts.
  - `README.md`: Documentation for synthetic forest fire experiments.
  - `NetLogo-Synthetic-Dataset-Generator/`: Directory for NetLogo synthetic data generation.
    - `README.md`: Detailed documentation for using the NetLogo synthetic dataset generator.
    - `pyNetLogo_script.py`: Script used for generating data (relevant for Figure 4).
  - `experiments/`: Contains various scripts for analyzing and evaluating synthetic data.
    - `stochastic_process_matrix_eval.py`
    - `ece_counterexample_study.py`
    - `reliability_analysis.py`
    - `predictive_difficulty_score_analysis.py`
    - `metric_evolution_vs_num_samples.py`
    - `calibration_curve_instant.py`

- **real_wildfire_dataset_experiments/**: Contains the real world wildfire data analysis scripts.
  - `README.md`: Documentation for real world wildfire experiments.
  - `experiments/`: Contains scripts for experiments on real world wildfire data.
    - `calibration_curve_baselines_compare.py`
    - `ece_score_table_generator.py`
