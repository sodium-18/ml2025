Here is a professional, academic-style **README.md** written in English. You can copy and paste this directly into your project folder.

It clearly explains the difference between the files, the datasets used, and the specific algorithmic improvements you implemented (Exponential Decay, Top-K, Smart Initialization).

---

# Enhanced D-TRUST: Robust Active Learning with Exponential Decay & Diversity Sampling

This project implements an optimized version of the D-TRUST Active Learning algorithm. It addresses the stability issues found in the original linear-decay approach by introducing **Exponential Decay** for hyperparameter adaptation and **Top-K Diversity Sampling** to avoid redundancy.

## 📂 Project Structure & File Description

### 1. Core Algorithm Modules

* **`sample.py` (Baseline)**
* **Description:** Contains the implementation of the original D-TRUST algorithm as described in the reference paper.
* **Mechanism:** Uses **Linear Decay** for hyperparameters \gamma (diversity) and \lambda (uncertainty).
* **Limitation:** The linear subtraction method often leads to negative weights in later iterations, causing severe performance oscillation (the "zigzag" effect).


* **`sample_final.py` (Ours / Improved)**
* **Description:** The optimized implementation containing our proposed contributions.
* **Key Modifications:**
1. **Exponential Decay:** Replaced linear subtraction with `np.exp(-decay_rate * t)`. This ensures weights remain strictly positive and converge smoothly, eliminating oscillation.
2. **Top-K Diversity (Deduplication):** Implemented a K-Means based filtering step within the sampling batch. It prevents the model from selecting multiple samples that are "hard" but identical (redundant information).
3. **Robustness:** Added safety checks to prevent crashes when the candidate pool is exhausted.





### 2. Experiment Execution Scripts

These scripts run comparative experiments (Ablation Studies) between the Baseline and the Proposed Method.

* **`run_yeast.py`**
* **Dataset:** **Yeast** (UCI).
* **Purpose:** Runs the primary comparison on a high-noise, difficult dataset.
* **Configuration:**
* **Baseline:** Random Initialization + Linear Decay (`sample.py`).
* **Ours:** Smart K-Means Initialization + Exponential Decay + Top-K (`sample_final.py`).


* **Output:** Generates `yeast_final_comparison.png` showing the learning curve gap.


* **`run_paper.py`**
* **Dataset:** **Dermatology** (UCI).
* **Purpose:** Demonstrates performance on a simpler, linearly separable dataset.
* **Configuration:** Compares Random Initialization (Baseline) vs. Smart Initialization (Ours).
* **Output:** Generates `dermatology_final_comparison.png`.


* **`run_medium.py` 
* **Dataset:** **Segment** or **Vehicle** (UCI).
* **Purpose:** Generic runners for medium-difficulty datasets to verify algorithmic generalization.
* **Features:** Includes moving average smoothing for visualization.



## 🚀 Key Improvements Summary

| Feature | Original D-TRUST (`sample.py`) | Improved D-TRUST (`sample_final.py`) |
| --- | --- | --- |
| **Weight Decay** | **Linear** (w_t = w_0 - \alpha t) <br>

<br> *Risk: Negative weights, oscillation.* | **Exponential** (w_t = w_0 e^{-\alpha t}) <br>

<br> *Benefit: Smooth convergence, stable.* |
| **Batch Sampling** | **Greedy** <br>

<br> *Risk: Selects redundant samples.* | **Top-K Diversity** <br>

<br> *Benefit: Selects representative & diverse samples.* |
| **Initialization** | **Random** <br>

<br> *Risk: Slow cold-start.* | **K-Means Clustering** <br>

<br> *Benefit: High-quality initial training set.* |

## 📊 How to Run

1. **Install Dependencies:**
```bash
pip install numpy pandas scipy scikit-learn matplotlib

```


2. **Run the Yeast Experiment:**
```bash
python run_yeast.py

```


3. **Run the Dermatology Experiment:**
```bash
python run_paper.py

```



## 📝 Dataset Info

* **Yeast:** 1484 instances, 8 attributes (High noise, Class overlap).
* **Dermatology:** 366 instances, 33 attributes (Clinical data).
* **Segment:** 2310 instances, 19 attributes (Image segmentation).
* **Vehicle:** 846 instances, 18 attributes (Silhouette classification).

*Note: All datasets are sourced from the UCI Machine Learning Repository.*