# Forest Fire Simulation Datasets

This folder contains two distinct datasets generated using the pyNetLogo script: `diffInit_dataset` and `sameInit_dataset`.

## Dataset Descriptions

### diffInit_dataset
The `diffInit_dataset` contains forest fire simulations with varying initial conditions for each run. Both the forest configuration and the fire seed location differ across simulations. This dataset is used for training Deep Neural Network (DNN) models and for reporting evaluation scores as discussed in the accompanying paper.

### sameInit_dataset
The `sameInit_dataset` contains simulations where the initial conditions are same across runs. The forest configuration and the fire seed location are identical for each simulation. This dataset represents the empirical stochastic process detailed in the paper.

## Dataset Structure

Both datasets share a similar directory structure:

```
root/
├── s-level_100: uniform_72_p_0
├── s-level_95: uniform_72_p_95
├── s-level_90: uniform_72_p_9
├── s-level_85: uniform_72_p_85
└── s-level_80: uniform_72_p_8
```

Within each `s-level` folder (e.g., `uniform_72_p_0`), the following subfolders are present:

- `Train/`
- `Test/`
- `Raw/`

### File Naming Convention

Files within the `Raw` folders follow the naming pattern:

```
experiment_<hash>_<video_index>_<frame_number>.png
```

These files are grouped by `<hash>_<video_index>` and stored as individual NumPy arrays in the `Train` and `Test` folders.

### Data Format

For the `Train` and `Test` sets, each video is saved as a NumPy array with the shape `(T, H, W, C)`:

- `T`: Number of frames in the video
- `H`: Height of each frame (in pixels)
- `W`: Width of each frame (in pixels)
- `C`: Number of color channels (3, corresponding to BGR)

## Usage

To utilize these datasets in your research or applications, ensure you have NumPy installed to handle the array formats. The images can be processed using standard image processing libraries compatible with PNG format.

