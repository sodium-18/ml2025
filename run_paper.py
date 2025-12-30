# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore")

# ==========================================
# [CORE: Import two versions of the algorithm]
# sample -> Original paper code (Baseline)
# sample_final -> Improved code (Enhanced)
# ==========================================
try:
    import sample as original_algo
    print("Successfully imported original algorithm: sample.py")
except ImportError:
    print("Error: sample.py not found")
    exit()

try:
    import sample_final as improved_algo
    print("Successfully imported improved algorithm: sample_final.py")
except ImportError:
    print("Error: sample_final.py not found")
    exit()

# === Smoothing function for visualization ===
def smooth_curve(points, weight=0.8):
    """
    Apply exponential moving average (EMA) smoothing for better visualization
    """
    last = points[0]
    smoothed = []
    for point in points:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def run_experiment(algo_module, name, X, y, train_idx, cand_idx, test_idx, M, is_improved=False):
    """Generic experiment runner function"""
    print(f"\n>>> Running: {name} ...")
    
    # Use paper-recommended SVM parameters
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1))
    
    # Deep copy indices to prevent modification of original lists
    current_train = list(train_idx)
    current_cand = list(cand_idx)
    
    if is_improved:
        # === Improved configuration ===
        # Enable Top-K deduplication (use_sv_dedup=True)
        # Exponential decay logic is built into sample_final
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.02, lam=1.0, lam_ur=0.02, gam_clf=0.1,
            use_sv_dedup=True, dedup_clusters=20 
        )
    else:
        # === Original configuration ===
        # Linear decay parameters
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.005, lam=1.0, lam_ur=0.01, gam_clf=0.1 
        )

    acc_list = [sampler.evaluate()]
    
    # Progress output
    for i in range(M):
        if len(sampler.candidate_index) == 0:
            print("  [Warning] Candidate pool exhausted, ending early")
            break
            
        try:
            sampler.dtrustSample(n_batch=1)
            acc = sampler.evaluate()
            acc_list.append(acc)
        except Exception as e:
            print(f"  [Error] Execution interrupted: {e}")
            break
            
        if (i+1) % 20 == 0:
            print(f"  Iter {i+1}/{M}: {acc:.4f}")
            
    return acc_list

def main():
    # 1. Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dermatology.data")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return

    print(f"Loading Dermatology dataset...")
    try:
        df = pd.read_csv(data_path, header=None)
    except Exception as e:
        print(f"Read error: {e}")
        return

    # 2. Preprocessing (Dermatology-specific)
    # Handle '?' missing values
    df.replace('?', np.nan, inplace=True)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Impute missing values (age column)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Label encoding & standardization
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Data overview: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # 3. Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    data_pool = np.vstack([X_train_full, X_test])
    labels_pool = np.concatenate([y_train_full, y_test])
    
    all_indices = np.arange(len(data_pool))
    test_start_idx = len(X_train_full)
    test_indices = list(all_indices[test_start_idx:])
    pool_indices = all_indices[:test_start_idx]
    
    # === Key: Fix random seed to ensure consistent starting point for both algorithms ===
    np.random.seed(42) 
    np.random.shuffle(pool_indices)
    
    # Initial set size: 10 samples
    init_train_indices = list(pool_indices[:10])
    init_cand_indices = list(pool_indices[10:])
    
    # Set iteration rounds
    M = 100
    print(f"Experiment setup: {M} iterations, 10 initial samples")

    # ==========================
    # 4. Run comparison experiment
    # ==========================
    
    # 4.1 Run original method
    acc_orig = run_experiment(
        original_algo, "Original (sample.py)",
        data_pool, labels_pool, 
        init_train_indices, init_cand_indices, test_indices, 
        M, is_improved=False
    )

    # 4.2 Run improved method
    acc_imp = run_experiment(
        improved_algo, "Improved (sample_final.py)",
        data_pool, labels_pool, 
        init_train_indices, init_cand_indices, test_indices, 
        M, is_improved=True
    )

    # ==========================
    # 5. Plot comparison results
    # ==========================
    plt.figure(figsize=(10, 6))
    
    # Generate smoothed curves for better visualization
    smooth_orig = smooth_curve(acc_orig, weight=0.75)
    smooth_imp = smooth_curve(acc_imp, weight=0.75)
    iters = range(len(acc_orig))

    # Plot raw data shadow (semi-transparent, indicating authenticity)
    plt.plot(iters, acc_orig, color='gray', alpha=0.2, linewidth=1)
    plt.plot(iters, acc_imp, color='red', alpha=0.15, linewidth=1)

    # Plot smoothed curves (main visualization)
    plt.plot(iters, smooth_orig, linestyle='--', color='gray', linewidth=2, label='Original D-TRUST')
    plt.plot(iters, smooth_imp, linestyle='-', color='#d62728', linewidth=2.5, label='Improved Method (Ours)')
    
    plt.title('Performance Comparison on Dermatology Dataset', fontsize=15)
    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(0.5, 1.0) # Dermatology accuracy typically high, lock range to see details
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    
    save_name = 'dermatology_compare_plot.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n[Complete] Comparison plot saved as: {save_name}")
    
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()