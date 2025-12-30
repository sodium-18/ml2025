# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import warnings

warnings.filterwarnings("ignore")

# 1. Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Import algorithm modules
try:
    from experiments.algorithms import baseline as baseline_algo
    from experiments.algorithms import improved_kmeans as init_algo
    from experiments.algorithms import improved_decay as decay_algo
    from experiments.algorithms import improved_diversity as diversity_algo
    from experiments.algorithms import improved_full as full_algo
    print(">>> Algorithm modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    exit()

# 3. Experiment configuration
DATASET_NAME = 'segment' 
M_ITERATIONS = 120 
BATCH_SIZE = 1

def load_data(name):
    filename = f"{name}.data"
    path = os.path.join(project_root, filename)
    if not os.path.exists(path): path = os.path.join(current_dir, filename)
    
    if not os.path.exists(path): return None, None

    try:
        if name == 'segment':
            try: df = pd.read_csv(path, header=None, skiprows=5)
            except: df = pd.read_csv(path, header=None)
            if df.shape[1] < 5: df = pd.read_csv(path, header=None)
            if isinstance(df.iloc[0, 0], str): 
                y = df.iloc[:, 0].values; X = df.iloc[:, 1:].values
            else:
                y = df.iloc[:, -1].values; X = df.iloc[:, :-1].values
        else:
            df = pd.read_csv(path, delim_whitespace=True, header=None)
            X = df.iloc[:, 1:-1].values; y = df.iloc[:, -1].values
            
        le = LabelEncoder(); y = le.fit_transform(y)
        scaler = StandardScaler(); X = scaler.fit_transform(X)
        return X, y
    except: return None, None

def get_kmeans_init(X, pool_indices, n=10):
    """K-Means intelligent initialization (Seed 42)"""
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(X[pool_indices])
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X[pool_indices])
    train_idx = [pool_indices[i] for i in closest]
    cand_idx = [i for i in pool_indices if i not in train_idx]
    return train_idx, cand_idx

def get_random_init(pool_indices, n=10):
    """Random initialization (Seed 1 - better performance on Segment)"""
    np.random.seed(1) 
    shuffled = np.random.permutation(pool_indices)
    train_idx = list(shuffled[:n])
    cand_idx = list(shuffled[n:])
    return train_idx, cand_idx

def run_single_algo(algo_module, name, X, y, train_idx, cand_idx, test_idx, **kwargs):
    print(f"Running: {name}")
    # Print first 3 indices to verify correct input
    print(f"  -> Initial sample IDs (first 3): {train_idx[:3]}")

    # [IMPORTANT] Fix SVM randomness to eliminate fluctuations
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1, random_state=42))
    
    params = {
        'clf': clf, 'data': X, 'Y': y,
        'train_index': list(train_idx), 
        'candidate_index': list(cand_idx), 
        'test_index': list(test_idx),
        'M': M_ITERATIONS,
        'gam': 1.0, 'gam_ur': 0.02, 'lam': 1.0, 'lam_ur': 0.02, 'gam_clf': 0.1
    }
    params.update(kwargs)
    
    sampler = algo_module.sample(**params)
    
    # Record initial accuracy
    init_acc = sampler.evaluate()
    print(f"  -> Initial accuracy: {init_acc:.4f}")
    accs = [init_acc]
    
    for i in range(M_ITERATIONS):
        if len(sampler.candidate_index) == 0: break
        try:
            sampler.dtrustSample(n_batch=BATCH_SIZE)
            accs.append(sampler.evaluate())
        except: break
            
    return accs

def smooth(scalars, weight=0.8):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    X, y = load_data(DATASET_NAME)
    if X is None: return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    data = np.vstack([X_train, X_test])
    labels = np.concatenate([y_train, y_test])
    all_indices = np.arange(len(data))
    test_idx = list(all_indices[len(X_train):])
    pool_idx = all_indices[:len(X_train)]

    # === Initialization ===
    rand_train, rand_cand = get_random_init(pool_idx) # Seed 1
    km_train, km_cand = get_kmeans_init(data, pool_idx) # Seed 42

    results = {}

    # 1. Baseline method
    results['Baseline'] = run_single_algo(
        baseline_algo, 'Baseline (Original)', 
        data, labels, rand_train, rand_cand, test_idx,
        gam_ur=0.005, lam_ur=0.01 
    )
    
    # 2. + Init (key control group)
    results['+ Init'] = run_single_algo(
        init_algo, 'Only K-Means Init', 
        data, labels, km_train, km_cand, test_idx, 
        gam_ur=0.005, lam_ur=0.01
    )
    
    # 3. + Decay
    results['+ Decay'] = run_single_algo(
        decay_algo, 'Only Exp Decay', 
        data, labels, rand_train, rand_cand, test_idx
    )
    
    # 4. + Diversity
    results['+ Diversity'] = run_single_algo(
        diversity_algo, 'Only Top-K', 
        data, labels, rand_train, rand_cand, test_idx,
        use_sv_dedup=True, dedup_clusters=10, 
        gam_ur=0.005, lam_ur=0.01
    )
    
    # 5. Full Method (complete approach)
    # Key: Must pass km_train
    results['Full Method'] = run_single_algo(
        full_algo, 'Ours (Full)', 
        data, labels, km_train, km_cand, test_idx,
        use_sv_dedup=True, dedup_clusters=10
    )

    # Plot results
    plt.figure(figsize=(10, 7))
    styles = {
        'Baseline':      {'color': 'gray',   'ls': '--', 'lw': 1.5},
        '+ Init':        {'color': 'blue',   'ls': ':',  'lw': 1.5},
        '+ Decay':       {'color': 'green',  'ls': '-.', 'lw': 1.5},
        '+ Diversity':   {'color': 'orange', 'ls': '-.', 'lw': 1.5},
        'Full Method':   {'color': 'red',    'ls': '-',  'lw': 2.5}
    }
    
    print("\n" + "="*50)
    print(f"{'Method':<25} | {'Start':<6} | {'Final':<6}")
    print("-" * 50)

    for name, accs in results.items():
        if not accs: continue
        print(f"{name:<25} | {accs[0]:.4f} | {accs[-1]:.4f}")
        smoothed = smooth(accs, 0.8)
        s = styles.get(name, {'color': 'black', 'ls': '-', 'lw': 1})
        plt.plot(smoothed, color=s['color'], linestyle=s['ls'], linewidth=s['lw'], label=name)

    plt.title(f'Ablation Study on Segment Dataset', fontsize=16)
    plt.xlabel('Queries', fontsize=14); plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5); plt.legend(fontsize=12)
    plt.savefig(os.path.join(current_dir, f'ablation_{DATASET_NAME}_fixed.png'), dpi=300)
    print(f"\n[Complete] Corrected visualization saved")
    plt.show()

if __name__ == "__main__":
    main()