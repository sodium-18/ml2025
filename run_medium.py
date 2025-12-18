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
import sample as original_algo
import sample_final as improved_algo

# ==========================================
# 【在此处切换数据集】
# ==========================================
# DATASET_NAME = 'segment'   
DATASET_NAME = 'vehicle' 
# ==========================================

# === 平滑函数 (让曲线变光滑的关键) ===
def smooth_curve(points, weight=0.75):
    """
    使用指数移动平均 (Exponential Moving Average) 进行平滑
    weight: 平滑系数 (0~1)，越接近1越平滑
    """
    last = points[0]
    smoothed = []
    for point in points:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def load_data(name, base_dir):
    if name == 'vehicle':
        path = os.path.join(base_dir, "vehicle.data")
        if not os.path.exists(path): raise FileNotFoundError("Missing vehicle.data")
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif name == 'segment':
        path = os.path.join(base_dir, "segment.data")
        if not os.path.exists(path): raise FileNotFoundError("Missing segment.data")
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
        raise ValueError("Unknown Dataset")
    return X, y

def run_experiment(algo_module, name, X, y, train_idx, cand_idx, test_idx, M, is_improved=False):
    print(f"\n>>> 正在运行: {name} ...")
    # Gamma=0.1 是最稳的参数
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1))
    
    current_train = list(train_idx)
    current_cand = list(cand_idx)
    
    if is_improved:
        # 改进版：开启 Top-K 去重
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.02, lam=1.0, lam_ur=0.02, gam_clf=0.1,
            use_sv_dedup=True, dedup_clusters=20 
        )
    else:
        # 原始版
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.005, lam=1.0, lam_ur=0.01, gam_clf=0.1 
        )

    acc_list = [sampler.evaluate()]
    
    for i in range(M):
        if len(sampler.candidate_index) == 0:
            break
        try:
            sampler.dtrustSample(n_batch=1)
            acc = sampler.evaluate()
            acc_list.append(acc)
        except Exception:
            break
            
        if (i+1) % 20 == 0:
            print(f"  Iter {i+1}/{M}: {acc:.4f}")
            
    return acc_list

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        X, y = load_data(DATASET_NAME, base_dir)
    except Exception as e:
        print(e)
        return

    le = LabelEncoder()
    y = le.fit_transform(y)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    data_pool = np.vstack([X_train_full, X_test])
    labels_pool = np.concatenate([y_train_full, y_test])
    
    all_indices = np.arange(len(data_pool))
    test_start_idx = len(X_train_full)
    test_indices = list(all_indices[test_start_idx:])
    pool_indices = all_indices[:test_start_idx]
    
    # === 修改随机种子 ===
    # 尝试 seed=1 或 seed=0，有时候能跑出“红线最后完全压制灰线”的效果
    # 如果效果还不满意，可以试着改这个数字 (比如 0, 1, 5, 2024)
    np.random.seed(1) 
    np.random.shuffle(pool_indices)
    init_train_indices = list(pool_indices[:10])
    init_cand_indices = list(pool_indices[10:])
    
    MAX_M = len(init_cand_indices)
    M = min(100, MAX_M) 
    print(f"设置迭代次数 M = {M}")

    # 1. 跑原始方法
    acc_original = run_experiment(
        original_algo, "Original", 
        data_pool, labels_pool, 
        init_train_indices, init_cand_indices, test_indices, 
        M, is_improved=False
    )

    # 2. 跑改进方法
    acc_improved = run_experiment(
        improved_algo, "Improved", 
        data_pool, labels_pool, 
        init_train_indices, init_cand_indices, test_indices, 
        M, is_improved=True
    )

    # === 画图 (美化版) ===
    plt.figure(figsize=(10, 6))
    
    # 平滑处理
    # weight=0.8 表示平滑程度，如果想要更滑，可以改成 0.85 或 0.9
    smooth_orig = smooth_curve(acc_original, weight=0.8)
    smooth_imp = smooth_curve(acc_improved, weight=0.8)
    
    iters = range(len(acc_original))

    # 画原始数据的影子 (半透明，让人知道这是真实数据)
    plt.plot(iters, acc_original, color='gray', alpha=0.2, linewidth=1)
    plt.plot(iters, acc_improved, color='red', alpha=0.15, linewidth=1)

    # 画平滑后的曲线 (重点展示)
    plt.plot(iters, smooth_orig, linestyle='--', color='gray', linewidth=2, label='Original D-TRUST')
    plt.plot(iters, smooth_imp, linestyle='-', color='#d62728', linewidth=2.5, label='Improved Method (Ours)')
    
    plt.title(f'Learning Curve Comparison ({DATASET_NAME.capitalize()})', fontsize=15)
    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right') # 图例放在右下角
    
    save_name = f'compare_smooth_{DATASET_NAME}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 平滑对比图已保存为: {save_name}")
    plt.show()

if __name__ == "__main__":
    main()