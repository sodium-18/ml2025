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

# 忽略一些 sklearn 的版本警告
warnings.filterwarnings("ignore")

# ==========================================
# 【核心：导入两个版本的算法】
# sample -> 原始论文代码 (Baseline)
# sample_final -> 你的改进代码 (Improved)
# ==========================================
try:
    import sample as original_algo
    print("成功导入原始算法: sample.py")
except ImportError:
    print("错误：找不到 sample.py")
    exit()

try:
    import sample_final as improved_algo
    print("成功导入改进算法: sample_final.py")
except ImportError:
    print("错误：找不到 sample_final.py")
    exit()

# === 平滑函数 (让曲线变光滑，适合论文展示) ===
def smooth_curve(points, weight=0.8):
    """
    使用指数移动平均 (Exponential Moving Average) 进行平滑
    """
    last = points[0]
    smoothed = []
    for point in points:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def run_experiment(algo_module, name, X, y, train_idx, cand_idx, test_idx, M, is_improved=False):
    """通用运行函数"""
    print(f"\n>>> 正在运行: {name} ...")
    
    # 使用论文推荐的 SVM 参数
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1))
    
    # 深度复制索引，防止修改原列表
    current_train = list(train_idx)
    current_cand = list(cand_idx)
    
    if is_improved:
        # === 改进版配置 ===
        # 开启 Top-K 去重 (use_sv_dedup=True)
        # 指数衰减逻辑内置在 sample_final 里
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.02, lam=1.0, lam_ur=0.02, gam_clf=0.1,
            use_sv_dedup=True, dedup_clusters=20 
        )
    else:
        # === 原始版配置 ===
        # 线性衰减参数
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.005, lam=1.0, lam_ur=0.01, gam_clf=0.1 
        )

    acc_list = [sampler.evaluate()]
    
    # 进度条打印
    for i in range(M):
        if len(sampler.candidate_index) == 0:
            print("  [警告] 候选池已空，提前结束")
            break
            
        try:
            sampler.dtrustSample(n_batch=1)
            acc = sampler.evaluate()
            acc_list.append(acc)
        except Exception as e:
            print(f"  [错误] 运行中断: {e}")
            break
            
        if (i+1) % 20 == 0:
            print(f"  Iter {i+1}/{M}: {acc:.4f}")
            
    return acc_list

def main():
    # 1. 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dermatology.data")

    if not os.path.exists(data_path):
        print(f"错误：没找到数据文件: {data_path}")
        return

    print(f"正在读取 Dermatology 数据...")
    try:
        df = pd.read_csv(data_path, header=None)
    except Exception as e:
        print(f"读取出错: {e}")
        return

    # 2. 预处理 (Dermatology 特有)
    # 处理 '?' 缺失值
    df.replace('?', np.nan, inplace=True)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 填补缺失值 (年龄列)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # 标签编码 & 标准化
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"数据概览: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")

    # 3. 划分数据
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    data_pool = np.vstack([X_train_full, X_test])
    labels_pool = np.concatenate([y_train_full, y_test])
    
    all_indices = np.arange(len(data_pool))
    test_start_idx = len(X_train_full)
    test_indices = list(all_indices[test_start_idx:])
    pool_indices = all_indices[:test_start_idx]
    
    # === 关键：固定随机种子，确保两个算法起跑线一致 ===
    np.random.seed(42) 
    np.random.shuffle(pool_indices)
    
    # 初始集大小：10个
    init_train_indices = list(pool_indices[:10])
    init_cand_indices = list(pool_indices[10:])
    
    # 设置迭代轮数
    M = 100
    print(f"实验设置: 迭代 {M} 轮, 初始样本 10 个")

    # ==========================
    # 4. 运行对比实验
    # ==========================
    
    # 4.1 跑原始方法
    acc_orig = run_experiment(
        original_algo, "Original (sample.py)", 
        data_pool, labels_pool, 
        init_train_indices, init_cand_indices, test_indices, 
        M, is_improved=False
    )

    # 4.2 跑改进方法
    acc_imp = run_experiment(
        improved_algo, "Improved (sample_final.py)", 
        data_pool, labels_pool, 
        init_train_indices, init_cand_indices, test_indices, 
        M, is_improved=True
    )

    # ==========================
    # 5. 绘制对比图
    # ==========================
    plt.figure(figsize=(10, 6))
    
    # 生成平滑曲线 (让图更好看)
    smooth_orig = smooth_curve(acc_orig, weight=0.75)
    smooth_imp = smooth_curve(acc_imp, weight=0.75)
    iters = range(len(acc_orig))

    # 画原始数据的影子 (半透明，表示真实性)
    plt.plot(iters, acc_orig, color='gray', alpha=0.2, linewidth=1)
    plt.plot(iters, acc_imp, color='red', alpha=0.15, linewidth=1)

    # 画平滑曲线 (主要展示)
    plt.plot(iters, smooth_orig, linestyle='--', color='gray', linewidth=2, label='Original D-TRUST')
    plt.plot(iters, smooth_imp, linestyle='-', color='#d62728', linewidth=2.5, label='Improved Method (Ours)')
    
    plt.title('Performance Comparison on Dermatology Dataset', fontsize=15)
    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.ylim(0.5, 1.0) # Dermatology 精度通常较高，锁定范围看细节
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    
    save_name = 'dermatology_compare_plot.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 对比图已保存为: {save_name}")
    
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()