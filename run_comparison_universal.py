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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import glob
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 【在此处切换数据集】
# 可选: 'dermatology', 'yeast', 'vehicle', 'segment'
# ==========================================
DATASET_NAME = 'dermatology'
# ==========================================

# 导入算法模块
try:
    import sample as original_algo
    print(">>> 成功导入原始算法: sample.py")
except ImportError:
    print("错误：找不到 sample.py")
    exit()

try:
    import sample_final as improved_algo
    print(">>> 成功导入改进算法: sample_final.py")
except ImportError:
    print("错误：找不到 sample_final.py")
    exit()

# === 平滑函数 ===
def smooth_curve(points, weight=0.8):
    """指数移动平均平滑"""
    last = points[0]
    smoothed = []
    for point in points:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# === 数据加载函数 (集成了所有数据集的处理逻辑) ===
def load_data(name, base_dir):
    print(f"正在读取 {name} 数据...")
    X, y = None, None
    
    if name == 'dermatology':
        path = os.path.join(base_dir, "dermatology.data")
        if not os.path.exists(path): raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path, header=None)
        df.replace('?', np.nan, inplace=True)
        # 填补缺失值
        imputer = SimpleImputer(strategy='mean')
        X_raw = df.iloc[:, :-1].values
        X = imputer.fit_transform(X_raw)
        y = df.iloc[:, -1].values
        
    elif name == 'yeast':
        path = os.path.join(base_dir, "yeast.data")
        if not os.path.exists(path): raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values
        
    elif name == 'vehicle':
        path = os.path.join(base_dir, "vehicle.data")
        # 自动处理 Vehicle 分散文件的情况
        if not os.path.exists(path):
            print("未找到 vehicle.data，尝试合并碎片文件...")
            part_files = sorted(glob.glob(os.path.join(base_dir, "x*.dat")))
            if part_files:
                with open(path, 'w') as outfile:
                    for fname in part_files:
                        with open(fname, 'r') as infile:
                            outfile.write(infile.read())
                print("合并完成！")
            else:
                raise FileNotFoundError("Missing vehicle.data and x*.dat files")
        
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
    elif name == 'segment':
        path = os.path.join(base_dir, "segment.data")
        if not os.path.exists(path): raise FileNotFoundError(f"Missing {path}")
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
        raise ValueError(f"未知数据集: {name}")
        
    return X, y

# === 运行实验核心逻辑 ===
def run_experiment(algo_module, name, X, y, train_idx, cand_idx, test_idx, M, is_improved=False):
    print(f"\n>>> 正在运行: {name} ...")
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1))
    
    current_train = list(train_idx)
    current_cand = list(cand_idx)
    
    if is_improved:
        # 改进版：Top-K 去重 + 指数衰减 (内置)
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.02, lam=1.0, lam_ur=0.02, gam_clf=0.1,
            use_sv_dedup=True, dedup_clusters=20 
        )
    else:
        # 原始版：无去重 + 线性衰减
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.005, lam=1.0, lam_ur=0.01, gam_clf=0.1 
        )

    acc_list = [sampler.evaluate()]
    
    for i in range(M):
        if len(sampler.candidate_index) == 0: break
        try:
            sampler.dtrustSample(n_batch=1)
            acc = sampler.evaluate()
            acc_list.append(acc)
        except Exception: break
            
        if (i+1) % 20 == 0:
            print(f"  Iter {i+1}/{M}: {acc:.4f}")
            
    return acc_list

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 加载并预处理数据
    try:
        X, y = load_data(DATASET_NAME, base_dir)
    except Exception as e:
        print(e)
        return

    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 2. 划分数据
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    data_pool = np.vstack([X_train_full, X_test])
    labels_pool = np.concatenate([y_train_full, y_test])
    all_indices = np.arange(len(data_pool))
    test_start_idx = len(X_train_full)
    test_indices = list(all_indices[test_start_idx:])
    pool_indices = all_indices[:test_start_idx]
    
    # ========================================================
    # 【核心逻辑：根据数据集类型决定初始化方式】
    # ========================================================
    
    # 默认：原始方法永远是随机 (Random)
    np.random.seed(42)
    shuffled_pool = np.random.permutation(pool_indices)
    orig_train_idx = list(shuffled_pool[:10])
    orig_cand_idx = list(shuffled_pool[10:])
    
    imp_train_idx = []
    imp_cand_idx = []
    
    # 策略判断
    if DATASET_NAME == 'dermatology':
        print("\n【模式】简单数据集: 强制双方都使用 Random 初始化 (公平起跑)")
        # 为了公平，且如你所说 dermatology 加上 K-Means 也没必要
        # 我们让改进版也用完全一样的随机种子起点
        imp_train_idx = list(orig_train_idx)
        imp_cand_idx = list(orig_cand_idx)
        init_method_str = "Random Init"
    else:
        print("\n【模式】复杂数据集: 改进版启用 Smart K-Means 初始化")
        # 其他数据集 (Yeast, Vehicle, Segment) 启用 K-Means 加速
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        kmeans.fit(data_pool[pool_indices])
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data_pool[pool_indices])
        imp_train_idx = [pool_indices[i] for i in closest]
        imp_cand_idx = [i for i in pool_indices if i not in imp_train_idx]
        init_method_str = "Smart Init (KMeans)"

    # 设置迭代次数 (Yeast 数据多，跑久一点；其他跑 100)
    M = 200 if DATASET_NAME == 'yeast' else 100
    print(f"实验设置: 迭代 {M} 轮")

    # 3. 运行实验
    # 3.1 原始方法
    acc_orig = run_experiment(
        original_algo, "Original (Baseline)", 
        data_pool, labels_pool, 
        orig_train_idx, orig_cand_idx, test_indices, 
        M, is_improved=False
    )

    # 3.2 改进方法
    acc_imp = run_experiment(
        improved_algo, f"Improved ({init_method_str})", 
        data_pool, labels_pool, 
        imp_train_idx, imp_cand_idx, test_indices, 
        M, is_improved=True
    )

    # 4. 画图 (美化版)
    plt.figure(figsize=(10, 6))
    
    smooth_orig = smooth_curve(acc_orig, weight=0.8)
    smooth_imp = smooth_curve(acc_imp, weight=0.8)
    iters = range(len(acc_orig))

    # 绘制阴影 (真实波动)
    plt.plot(iters, acc_orig, color='gray', alpha=0.2, linewidth=1)
    plt.plot(iters, acc_imp, color='red', alpha=0.15, linewidth=1)

    # 绘制平滑曲线 (主视觉)
    plt.plot(iters, smooth_orig, linestyle='--', color='gray', linewidth=2, label='Original D-TRUST')
    plt.plot(iters, smooth_imp, linestyle='-', color='#d62728', linewidth=2.5, label='Improved Method (Ours)')
    
    plt.title(f'Performance Comparison on {DATASET_NAME.capitalize()} Dataset', fontsize=15)
    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    
    # 针对不同数据集微调 Y 轴，让图更好看
    if DATASET_NAME == 'dermatology':
        plt.ylim(0.5, 1.0)
    elif DATASET_NAME == 'yeast':
        plt.ylim(0.4, 0.65) # Yeast 很难，通常分不高
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    
    save_name = f'compare_final_{DATASET_NAME}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 对比图已保存为: {save_name}")
    plt.show()

if __name__ == "__main__":
    main()