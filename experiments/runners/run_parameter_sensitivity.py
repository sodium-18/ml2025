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

# 1. 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入动态参数模块
try:
    from experiments.algorithms import improved_full1 as flexible_algo
    print(">>> 成功导入动态参数模块: improved_full1.py")
except ImportError:
    print("错误：找不到 improved_full1.py")
    exit()

# 3. 基础配置
DATASET_NAME = 'segment' 
M_ITERATIONS = 100  # 强制所有实验跑 100 轮
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
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(X[pool_indices])
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X[pool_indices])
    train_idx = [pool_indices[i] for i in closest]
    cand_idx = [i for i in pool_indices if i not in train_idx]
    return train_idx, cand_idx

def run_algo(X, y, train_idx, cand_idx, test_idx, decay_rate=0.02, dedup_k=20, label="", custom_batch=1):
    print(f"  -> Running: {label} (Batch={custom_batch}, Rounds={M_ITERATIONS})")
    
    # 锁定 SVM 随机性
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1, random_state=42))
    
    sampler = flexible_algo.sample(
        clf=clf, data=X, Y=y,
        train_index=list(train_idx), candidate_index=list(cand_idx), test_index=list(test_idx),
        M=M_ITERATIONS,
        gam_ur=decay_rate,       
        dedup_clusters=dedup_k, 
        use_sv_dedup=True
    )
    
    accs = [sampler.evaluate()]
    
    # 强制跑满 M_ITERATIONS 轮
    for i in range(M_ITERATIONS):
        # 只有当候选池不够时才停止
        if len(sampler.candidate_index) < custom_batch: 
            print(f"    [Warning] Pool exhausted at round {i}")
            break
        try:
            sampler.dtrustSample(n_batch=custom_batch)
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

def plot_experiment(results, title, filename):
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    styles = ['-', '--', '-.']
    
    for i, (label, accs) in enumerate(results.items()):
        smoothed = smooth(accs, 0.8)
        # 确保只绘制到100轮
        smoothed = smoothed[:M_ITERATIONS+1]
        
        c = colors[i % len(colors)]
        s = styles[i % len(styles)]
        
        # 图例只显示参数名，不显示最终精度（太长了）
        plt.plot(smoothed, label=label, color=c, linestyle=s, linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Rounds', fontsize=12) # 明确X轴为 Rounds
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=11, loc='lower right')
    
    save_path = os.path.join(current_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[完成] 图片已保存: {save_path}\n")

def main():
    X, y = load_data(DATASET_NAME)
    if X is None: return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    data = np.vstack([X_train, X_test])
    labels = np.concatenate([y_train, y_test])
    all_indices = np.arange(len(data))
    test_idx = list(all_indices[len(X_train):])
    pool_idx = all_indices[:len(X_train)]

    print(f"=== 参数敏感性实验 (Segment) - 100 Rounds ===")

    # -------------------------------------------------
    # Exp 1: Initialization Size (Init K)
    # -------------------------------------------------
    print("\n>>> Exp 1: Init Size (Batch=1, 100 Rounds)")
    results_init = {}
    init_k_values = [5, 10, 20] 
    for k in init_k_values:
        t_idx, c_idx = get_kmeans_init(data, pool_idx, n=k)
        # 固定 Decay=0.02, TopK=10
        accs = run_algo(data, labels, t_idx, c_idx, test_idx, decay_rate=0.02, dedup_k=10, label=f"Init K={k}", custom_batch=1)
        results_init[f"Init K={k}"] = accs
    plot_experiment(results_init, "Effect of Initialization Size (K-Means)", "sensitivity_init.png")

    # -------------------------------------------------
    # Exp 2: Decay Rate (Alpha)
    # -------------------------------------------------
    print(">>> Exp 2: Decay Rate (Batch=1, 100 Rounds)")
    results_decay = {}
    # 【更新】您指定的数值
    decay_values = [0.005, 0.02, 0.1]
    base_t_idx, base_c_idx = get_kmeans_init(data, pool_idx, n=10)
    
    for rate in decay_values:
        accs = run_algo(data, labels, base_t_idx, base_c_idx, test_idx, decay_rate=rate, dedup_k=10, label=f"Decay={rate}", custom_batch=1)
        results_decay[f"Decay={rate}"] = accs
    plot_experiment(results_decay, "Effect of Decay Rate", "sensitivity_decay.png")

    # -------------------------------------------------
    # Exp 3: Top-K Diversity
    # 注意：Batch=5 才能看出 Top-K 的区别，但我们现在强制跑满 100 轮
    # -------------------------------------------------
    print(">>> Exp 3: Top-K Diversity (Batch=5, 100 Rounds)")
    results_topk = {}
    # 选用 5, 20, 50 以拉开差距
    topk_values = [5, 20, 50] 
    
    for k in topk_values:
        # custom_batch=5: 这样一次选5个，Top-K去重逻辑才会生效
        accs = run_algo(data, labels, base_t_idx, base_c_idx, test_idx, decay_rate=0.02, dedup_k=k, label=f"Top-K={k}", custom_batch=5)
        results_topk[f"Top-K={k}"] = accs
    plot_experiment(results_topk, "Effect of Diversity Strength (Top-K)", "sensitivity_topk.png")

if __name__ == "__main__":
    main()
    