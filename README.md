# AID-TRUST: Improved Active Learning with Adaptive Decay and Smart Initialization

## Project Overview

This project implements AID-TRUST (Adaptive Initialized Diversity - TRUST), an enhanced active learning framework based on the D-TRUST algorithm.

The primary goal of this project is to address specific limitations in traditional active learning methods, such as the "cold-start" problem caused by random initialization and the rigidity of linear parameter decay. By introducing K-Means initialization, exponential weight decay, and Top-K diversity sampling, the proposed method achieves higher sample efficiency and classification accuracy on UCI benchmark datasets.

## Key Improvements

Compared to the baseline (original D-TRUST), this implementation introduces three main contributions:

1. Smart Initialization (K-Means):Replaces random sampling with K-Means clustering to select the initial training set. This ensures better geometric coverage of the data manifold and mitigates the cold-start problem.
2. Adaptive Exponential Decay:Replaces linear decay with an exponential decay function for the trade-off parameters ( and ). This allows the model to transition more smoothly from exploration (diversity) to exploitation (uncertainty).
3. Top-K Diversity Deduplication:Implements a Top-K selection mechanism in the evidence space (gradient space) to filter out redundant samples that are too similar to the current support vectors.

## Repository Structure

### Core Algorithms

`sample_final.py`: **The proposed method (AID-TRUST).** Contains the implementation of K-Means initialization, exponential decay, and Top-K clustering logic.
`sample_old.py`: **The baseline method.** Contains the original D-TRUST implementation with random initialization and linear decay for comparison.

### Main Execution Scripts

`run_medium.py`: The main entry point for running a quick comparison between the Baseline and AID-TRUST on the Vehicle dataset. Generates the "Red vs. Grey" comparison plot.
`run_comparison_universal.py`: A unified script to run experiments across multiple datasets (Yeast, Segmentation, Vehicle, Dermatology).
`run_yeast.py` / `run_paper.py`: Specific runner scripts for different datasets or experimental settings.

### Experiments & Analysis (`experiments/runners/`)

 `run_abla.py`: Runs the ablation study to verify the contribution of each module (Init, Decay, Diversity) separately.
 `run_parameter_sensitivity.py`: Tests the sensitivity of hyperparameters (e.g., Decay Rate, Top-K size, Initialization size).

### Data

The project uses standard UCI Machine Learning Repository datasets processed into `.data` or `.csv` format:

`vehicle.data`
`segment.data`
`yeast.data`
`dermatology.data`

## Requirements

 Python 3.8+
 NumPy
 Pandas
 Scikit-learn
 Matplotlib
 SciPy

You can install the dependencies via pip:

```bash
pip install numpy pandas scikit-learn matplotlib scipy

```

## Usage

### 1. Run the Main Comparison

To see the performance comparison between the proposed method and the baseline on the Vehicle dataset:

```bash
python run_medium.py

```

This will generate a plot showing the learning curves (Accuracy vs. Number of Queries).

To run on all datasets:

```bash
python experiments/runners/run_comparison_universal.py

```

### 2. Run Ablation Studies

To analyze the effect of each individual component (e.g., only changing initialization or only changing decay):

```bash
python experiments/runners/run_abla.py

```

### 3. Run Parameter Sensitivity Analysis

To test how different parameters (like decay rates 0.005 vs 0.1) affect performance:

```bash
python experiments/runners/run_parameter_sensitivity.py

```

### 4. Running the Executable Demo

If you have the compiled `.exe` file (`AID-TRUST_Demo.exe`):

1. Ensure the data file (e.g., `vehicle.data`) is in the same directory (or bundled within the exe).
2. Double-click to run.
3. The program will execute the active learning loop and display the result plot automatically.

## Experimental Results

Experiments were conducted on Vehicle, Segment, Yeast, and Dermatology datasets.

* **Convergence:** The exponential decay strategy allows the model to improve accuracy faster in the early-to-mid stages .
* **Stability:** Top-K deduplication prevents performance drops in later stages by avoiding redundant sampling.
* **Initialization:** K-Means initialization provides a higher starting accuracy (approx. +5% to +10%) compared to random initialization.

## License

This project is for academic and research purposes.