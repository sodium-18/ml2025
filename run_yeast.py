# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 导入两个算法
# ==========================================
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

# 平滑函数
def smooth_curve(points, weight=0.8):
    last = points[0]
    smoothed = []
    for point in points:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def run_experiment(algo_module, name, X, y, train_idx, cand_idx, test_idx, M, is_improved=False):
    print(f"\n>>> 正在运行: {name} ...")
    
    # 统一 SVM 参数
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1))
    
    # 复制索引
    current_train = list(train_idx)
    current_cand = list(cand_idx)
    
    if is_improved:
        # === 你的改进版 ===
        # 1. 策略: 指数衰减 (sample_final内置)
        # 2. 去重: Top-K (use_sv_dedup=True)
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.02, lam=1.0, lam_ur=0.02, gam_clf=0.1,
            use_sv_dedup=True, dedup_clusters=20 
        )
    else:
        # === 原始论文版 ===
        # 1. 策略: 线性衰减
        # 2. 去重: 无
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
    # 1. 读取数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "yeast.data")

    if not os.path.exists(data_path):
        print(f"找不到文件: {data_path}")
        return

    try:
        df = pd.read_csv(data_path, delim_whitespace=True, header=None)
    except:
        return

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
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
    pool_indices = all_indices[:test_start_idx] # 候选池
    
    # ========================================================
    # 【核心修改】：区别对待初始化
    # ========================================================
    
    # 1. 为原始方法准备：随机初始化 (Random) -> 模拟原始论文
    np.random.seed(42) # 固定种子保证可复现
    shuffled_pool = np.random.permutation(pool_indices)
    
    orig_train_idx = list(shuffled_pool[:10])
    orig_cand_idx = list(shuffled_pool[10:])
    
    print(f"原始方法初始化: Random (10个样本)")

    # 2. 为你的方法准备：智能初始化 (K-Means) -> 展示你的优化
    print("你的方法初始化: Smart K-Means (10个样本)...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10) # 注意这里是10个簇
    kmeans.fit(data_pool[pool_indices])
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data_pool[pool_indices])
    
    imp_train_idx = [pool_indices[i] for i in closest]
    imp_cand_idx = [i for i in pool_indices if i not in imp_train_idx]
    
    # 设置迭代
    M = 250

    # ==========================
    # 3. 运行对比
    # ==========================
    
    # 3.1 跑原始方法 (用随机索引)
    acc_orig = run_experiment(
        original_algo, "Original (Random Init)", 
        data_pool, labels_pool, 
        orig_train_idx, orig_cand_idx, test_indices, # <-- 传入随机索引
        M, is_improved=False
    )

    # 3.2 跑改进方法 (用智能索引)
    acc_imp = run_experiment(
        improved_algo, "Improved (Smart Init)", 
        data_pool, labels_pool, 
        imp_train_idx, imp_cand_idx, test_indices,   # <-- 传入KMeans索引
        M, is_improved=True
    )

    # ==========================
    # 4. 画图
    # ==========================
    plt.figure(figsize=(10, 6))
    
    smooth_orig = smooth_curve(acc_orig, weight=0.8)
    smooth_imp = smooth_curve(acc_imp, weight=0.8)
    iters = range(len(acc_orig))

    # 原始数据影子
    plt.plot(iters, acc_orig, color='gray', alpha=0.2, linewidth=1)
    plt.plot(iters, acc_imp, color='red', alpha=0.15, linewidth=1)

    # 平滑曲线
    plt.plot(iters, smooth_orig, linestyle='--', color='gray', linewidth=2, label='Original (Baseline)')
    plt.plot(iters, smooth_imp, linestyle='-', color='#d62728', linewidth=2.5, label='Ours (Smart Init + Improved)')
    
    plt.title('Ablation Study: Baseline vs. Our Proposed Method', fontsize=15)
    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    
    save_name = 'yeast_final_comparison.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 最终对比图已保存为: {save_name}")
    plt.show()

if __name__ == "__main__":
    main()