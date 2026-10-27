# t-route and Kalman Filter Interface

The `Interface` class connects t-route's Basic Model Interface (BMI) with the Kalman filter (KF) model provided by `tx-fast-hydrology`. It coordinates model initialization, exchanges routing parameters and discharge values, and supports warmup and forecast runs.

## Installation

First, complete the [t-route installation](../readme.md). Then install the additional KF dependency into the same Python environment:

```bash
uv pip install "tx-fast-hydrology @ git+https://github.com/slama0077/tx-fast-hydrology_SL.git@EKF_Troute"
```

## Configuration and usage

Create an `Interface` instance with paths to the t-route and KF YAML configuration files. Each model initializes from its configuration file's directory, so relative input paths should be defined accordingly. Both configurations must use the same time step and start datetime; initialization raises an error if they differ.

See [interface.ipynb](interface.ipynb) for an example workflow and [Interface.py](Interface.py) for the implementation. The `KF_Files/` directory contains supporting KF configuration and input files.

The main methods are:

- `update()`: Advance t-route, pass routing parameters and discharge to the KF model, and feed the resulting discharge back into t-route for the next step.
- `warmup_with_KF(num_steps=96)`: Run the coupled models for the requested number of warmup steps.
- `warmup_without_KF(num_steps=96)`: Advance only t-route during warmup.
- `forecast_only(num_steps=96)`: Run a forecast using a copy of the current t-route model and record discharge at the starting time and each subsequent step.
- `get_results_dataframe()`: Return recorded forecast discharge as a pandas DataFrame, with timestamps as rows and reach IDs as columns. Returns an empty DataFrame if no forecast has been recorded.
