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
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"项目根目录设置为: {project_root}")

# ==========================================
# 2. 导入算法模块
# ==========================================
try:
    from experiments.algorithms import baseline as baseline_algo
    from experiments.algorithms import improved_kmeans as init_algo
    from experiments.algorithms import improved_decay as decay_algo
    from experiments.algorithms import improved_diversity as diversity_algo
    # from experiments.algorithms import improved_full as full_algo

    print(">>> 所有算法模块导入成功！")
except ImportError as e:
    print(f"导入错误: {e}")
    exit()

# ==========================================
# 3. 核心配置
# ==========================================
DATASET_NAME = 'segment'
M_ITERATIONS = 120
BATCH_SIZE = 1


def load_data(name):
    """加载数据"""
    filename = f"{name}.data"
    path = os.path.join(project_root, filename)

    if not os.path.exists(path):
        path = os.path.join(current_dir, filename)

    if not os.path.exists(path):
        print(f"错误: 找不到数据文件 {filename}")
        return None, None

    print(f"正在加载数据: {path}")

    try:
        if name == 'segment':
            try:
                df = pd.read_csv(path, header=None, skiprows=5)
                if df.shape[1] < 5: df = pd.read_csv(path, header=None)
            except:
                df = pd.read_csv(path, header=None)

            if isinstance(df.iloc[0, 0], str):
                y = df.iloc[:, 0].values
                X = df.iloc[:, 1:].values
            else:
                y = df.iloc[:, -1].values
                X = df.iloc[:, :-1].values
        else:
            df = pd.read_csv(path, delim_whitespace=True, header=None)
            X = df.iloc[:, 1:-1].values
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
    """K-Means 智能初始化"""
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(X[pool_indices])
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X[pool_indices])
    train_idx = [pool_indices[i] for i in closest]
    cand_idx = [i for i in pool_indices if i not in train_idx]
    return train_idx, cand_idx


def get_random_init(pool_indices, n=10):
    """随机初始化"""
    np.random.seed(42)
    shuffled = np.random.permutation(pool_indices)
    train_idx = list(shuffled[:n])
    cand_idx = list(shuffled[n:])
    return train_idx, cand_idx


# ==========================================
# 核心修复: run_single_algo
# ==========================================
def run_single_algo(algo_module, name, X, y, train_idx, cand_idx, test_idx, **kwargs):
    print(f"正在运行: {name} ...")

    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1))

    # 1. 定义默认参数字典
    params = {
        'clf': clf,
        'data': X,
        'Y': y,
        'train_index': list(train_idx),
        'candidate_index': list(cand_idx),
        'test_index': list(test_idx),
        'M': M_ITERATIONS,
        'gam': 1.0,
        'gam_ur': 0.02,  # 默认衰减率
        'lam': 1.0,
        'lam_ur': 0.02,  # 默认衰减率
        'gam_clf': 0.1
    }

    # 2. 用传入的 kwargs 覆盖默认参数 (关键修复！)
    # 这样 Baseline 传入的 gam_ur=0.005 就会覆盖掉默认的 0.02
    params.update(kwargs)

    # 3. 将字典解包传入
    sampler = algo_module.sample(**params)

    accs = [sampler.evaluate()]

    for i in range(M_ITERATIONS):
        if len(sampler.candidate_index) == 0:
            break

        try:
            sampler.dtrustSample(n_batch=BATCH_SIZE)
            accs.append(sampler.evaluate())
        except Exception as e:
            print(f"  [{name}] 运行中断: {e}")
            break

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
    test_start = len(X_train)
    test_idx = list(all_indices[test_start:])
    pool_idx = all_indices[:test_start]

    print(f"数据准备完毕: {len(pool_idx)} 候选样本, {len(test_idx)} 测试样本")

    rand_train, rand_cand = get_random_init(pool_idx)
    km_train, km_cand = get_kmeans_init(data, pool_idx)

    results = {}

    # 1. Baseline (原始 D-TRUST)
    results['Baseline'] = run_single_algo(
        baseline_algo, 'Baseline (Original)',
        data, labels, rand_train, rand_cand, test_idx,
        gam_ur=0.005, lam_ur=0.01  # 这里传入的值会覆盖默认值
    )

    # 2. + Init
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
        use_sv_dedup=True, dedup_clusters=20,
        gam_ur=0.005, lam_ur=0.01
    )

    # # 5. Full Method
    # results['Full Method'] = run_single_algo(
    #     full_algo, 'Ours (Full)',
    #     data, labels, km_train, km_cand, test_idx,
    #     use_sv_dedup=True, dedup_clusters=20
    # )

    plt.figure(figsize=(12, 8))

    styles = {
        'Baseline': {'color': 'gray', 'ls': '--', 'lw': 1.5, 'label': 'Baseline (Linear, Random)'},
        '+ Init': {'color': 'blue', 'ls': ':', 'lw': 1.5, 'label': '+ Smart Init (K-Means)'},
        '+ Decay': {'color': 'green', 'ls': '-.', 'lw': 1.5, 'label': '+ Exp Decay'},
        '+ Diversity': {'color': 'orange', 'ls': '-.', 'lw': 1.5, 'label': '+ Top-K Diversity'}
        # 'Full Method': {'color': 'red', 'ls': '-', 'lw': 2.5, 'label': 'Ours (Full Method)'}
    }

    print("\n" + "=" * 40)
    print(f"{'Method':<25} | {'Final Acc':<10}")
    print("-" * 40)

    for name, accs in results.items():
        if not accs: continue
        print(f"{name:<25} | {accs[-1]:.4f}")
        smoothed_acc = smooth(accs, 0.8)
        s = styles.get(name, {'color': 'black', 'ls': '-', 'lw': 1, 'label': name})
        plt.plot(range(len(smoothed_acc)), smoothed_acc,
                 color=s['color'], linestyle=s['ls'], linewidth=s['lw'], label=s['label'])

    plt.title(f'Ablation Study on {DATASET_NAME.capitalize()} Dataset', fontsize=16)
    plt.xlabel('Number of Rounds', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='lower right')

    save_path = os.path.join(current_dir, f'ablation_{DATASET_NAME}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 消融实验图已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()