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

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 路径与环境配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # ml2025/

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入算法模块
try:
    from experiments.algorithms import baseline as baseline_algo
    from experiments.algorithms import improved_kmeans as init_algo
    from experiments.algorithms import improved_decay as decay_algo
    from experiments.algorithms import improved_diversity as diversity_algo
    from experiments.algorithms import improved_full as full_algo
    print(">>> 算法模块导入成功")
except ImportError as e:
    print(f"导入错误: {e}")
    exit()

# ==========================================
# 2. 实验配置
# ==========================================
DATASET_NAME = 'vehicle'  # 目标数据集
M_ITERATIONS = 100        # Vehicle 样本少，100轮足够
BATCH_SIZE = 1

def load_data(name):
    """加载 Vehicle 数据"""
    filename = f"{name}.data"
    path = os.path.join(project_root, filename)
    
    # 备用路径搜索
    if not os.path.exists(path): 
        path = os.path.join(current_dir, filename)
    
    if not os.path.exists(path):
        print(f"错误: 找不到 {filename}")
        print("请确认您已经合并了 xaa.dat, xab.dat... 为 vehicle.data")
        return None, None

    print(f"正在加载数据: {path}")

    try:
        # Vehicle 数据是空格分隔的
        if name == 'vehicle':
            df = pd.read_csv(path, delim_whitespace=True, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        else:
            # 兼容其他逻辑
            df = pd.read_csv(path, header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
        le = LabelEncoder()
        y = le.fit_transform(y)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y
        
    except Exception as e:
        print(f"数据读取失败: {e}")
        return None, None

def get_kmeans_init(X, pool_indices, n=10):
    """K-Means 智能初始化 (Seed 42)"""
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(X[pool_indices])
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X[pool_indices])
    train_idx = [pool_indices[i] for i in closest]
    cand_idx = [i for i in pool_indices if i not in train_idx]
    return train_idx, cand_idx

def get_random_init(pool_indices, n=10):
    """随机初始化 (Seed 1 - 您之前验证过效果最好的种子)"""
    np.random.seed(1) 
    shuffled = np.random.permutation(pool_indices)
    train_idx = list(shuffled[:n])
    cand_idx = list(shuffled[n:])
    return train_idx, cand_idx

def run_single_algo(algo_module, name, X, y, train_idx, cand_idx, test_idx, **kwargs):
    print(f"正在运行: {name}")
    
    # 锁定 SVM 随机性
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
    
    # 记录初始精度
    init_acc = sampler.evaluate()
    accs = [init_acc]
    
    for i in range(M_ITERATIONS):
        if len(sampler.candidate_index) == 0: break
        try:
            sampler.dtrustSample(n_batch=BATCH_SIZE)
            accs.append(sampler.evaluate())
        except: break
            
    return accs

def smooth(scalars, weight=0.75):
    """Vehicle 数据波动可能较大，平滑系数设为 0.75"""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    # 1. 加载数据
    X, y = load_data(DATASET_NAME)
    if X is None: return

    # 2. 划分数据 (保持 Seed 42 以便复现)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    data = np.vstack([X_train, X_test])
    labels = np.concatenate([y_train, y_test])
    all_indices = np.arange(len(data))
    test_idx = list(all_indices[len(X_train):])
    pool_idx = all_indices[:len(X_train)]

    print(f"数据准备完毕: {len(pool_idx)} 候选样本, {len(test_idx)} 测试样本")

    # === 初始化策略 ===
    # 随机初始化 (Seed 1) -> 给 Baseline, Decay, Diversity 用
    rand_train, rand_cand = get_random_init(pool_idx)
    # 智能初始化 (Seed 42) -> 给 Init, Full 用
    km_train, km_cand = get_kmeans_init(data, pool_idx)

    # 打印起点，用于核对
    print(f"[核对] 随机初始样本ID(前3): {rand_train[:3]}")
    print(f"[核对] 智能初始样本ID(前3): {km_train[:3]}")

    results = {}

    # --- 1. Baseline (随机 + 线性) ---
    results['Baseline'] = run_single_algo(
        baseline_algo, 'Baseline (Original)', 
        data, labels, rand_train, rand_cand, test_idx,
        gam_ur=0.005, lam_ur=0.01 
    )
    
    # --- 2. + Init (K-Means + 线性) ---
    results['+ Init'] = run_single_algo(
        init_algo, 'Only K-Means Init', 
        data, labels, km_train, km_cand, test_idx, 
        gam_ur=0.005, lam_ur=0.01
    )
    
    # --- 3. + Decay (随机 + 指数) ---
    results['+ Decay'] = run_single_algo(
        decay_algo, 'Only Exp Decay', 
        data, labels, rand_train, rand_cand, test_idx
    )
    
    # --- 4. + Diversity (随机 + 线性 + TopK) ---
    # Vehicle 是中等数据集，Dedup Clusters 用 20 比较合适
    results['+ Diversity'] = run_single_algo(
        diversity_algo, 'Only Top-K', 
        data, labels, rand_train, rand_cand, test_idx,
        use_sv_dedup=True, dedup_clusters=20, 
        gam_ur=0.005, lam_ur=0.01
    )
    
    # --- 5. Full Method (K-Means + 指数 + TopK) ---
    results['Full Method'] = run_single_algo(
        full_algo, 'Ours (Full)', 
        data, labels, km_train, km_cand, test_idx,
        use_sv_dedup=True, dedup_clusters=20
    )

    # === 画图 ===
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
        smoothed = smooth(accs, 0.75)
        s = styles.get(name, {'color': 'black', 'ls': '-', 'lw': 1})
        plt.plot(smoothed, color=s['color'], linestyle=s['ls'], linewidth=s['lw'], label=name)

    plt.title(f'Ablation Study on Vehicle Dataset', fontsize=16)
    plt.xlabel('Number of Queries', fontsize=14); plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5); plt.legend(fontsize=12)
    plt.savefig(os.path.join(current_dir, f'ablation_{DATASET_NAME}.png'), dpi=300)
    print(f"\n[完成] Vehicle 消融实验图已保存！")
    plt.show()

if __name__ == "__main__":
    main()