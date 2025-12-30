# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import json
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
from datetime import datetime
from pathlib import Path

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 【在此处切换数据集】
# 可选: 'dermatology', 'yeast', 'vehicle', 'segment'
# ==========================================
DATASET_NAME = 'yeast'  # 修改此行以选择数据集
# ==========================================

# 导入算法模块
try:
    import sample_old as original_algo
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

# === 配置 ===
class ExperimentConfig:
    """实验配置"""
    def __init__(self):
        self.dataset_name = DATASET_NAME
        self.random_state = 42
        self.test_size = 0.3
        self.smooth_weight = 0.8
        self.save_results = True
        self.results_dir = "experiment_results"
        
    def get_experiment_id(self):
        """生成实验ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.dataset_name}_{timestamp}"

# === 实验结果 ===
class ExperimentResult:
    """存储单次实验结果"""
    def __init__(self, config):
        self.config = config.__dict__.copy()
        self.experiment_id = config.get_experiment_id()
        self.timestamp = datetime.now().isoformat()
        self.original_acc = []
        self.improved_acc = []
        self.original_smooth = []
        self.improved_smooth = []
        self.init_method = ""
        self.dataset_stats = {}
        self.runtime_info = {}
        
    def to_dict(self):
        """转换为字典"""
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'dataset': self.config['dataset_name'],
            'original_acc': self.original_acc,
            'improved_acc': self.improved_acc,
            'original_smooth': self.original_smooth,
            'improved_smooth': self.improved_smooth,
            'init_method': self.init_method,
            'dataset_stats': self.dataset_stats,
            'runtime_info': self.runtime_info
        }
    
    def save(self, filename):
        """保存结果到文件"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"实验结果已保存至: {filename}")

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

# === 数据加载函数 ===
def load_data(name, base_dir):
    print(f"正在读取 {name} 数据...")
    X, y = None, None
    
    if name == 'dermatology':
        path = os.path.join(base_dir, "dermatology.data")
        if not os.path.exists(path): raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path, header=None)
        df.replace('?', np.nan, inplace=True)
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
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M, 
            gam=1.0, gam_ur=0.02, lam=1.0, lam_ur=0.02, gam_clf=0.1,
            use_sv_dedup=True, dedup_clusters=20 
        )
    else:
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

# === 运行完整实验 ===
def run_full_experiment():
    """运行完整的实验流程"""
    print(f"\n{'='*60}")
    print(f"开始实验: {DATASET_NAME}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    config = ExperimentConfig()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 加载并预处理数据
    try:
        X, y = load_data(config.dataset_name, base_dir)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 2. 划分数据
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    data_pool = np.vstack([X_train_full, X_test])
    labels_pool = np.concatenate([y_train_full, y_test])
    all_indices = np.arange(len(data_pool))
    test_start_idx = len(X_train_full)
    test_indices = list(all_indices[test_start_idx:])
    pool_indices = all_indices[:test_start_idx]
    
    # 3. 初始化设置
    np.random.seed(config.random_state)
    shuffled_pool = np.random.permutation(pool_indices)
    orig_train_idx = list(shuffled_pool[:10])
    orig_cand_idx = list(shuffled_pool[10:])
    
    init_method_str = "Random Init"
    imp_train_idx = []
    imp_cand_idx = []
    
    if config.dataset_name == 'dermatology':
        print("\n【模式】简单数据集: 双方都使用 Random 初始化")
        imp_train_idx = list(orig_train_idx)
        imp_cand_idx = list(orig_cand_idx)
    else:
        print("\n【模式】复杂数据集: 改进版启用 Smart K-Means 初始化")
        kmeans = KMeans(n_clusters=10, random_state=config.random_state, n_init=10)
        kmeans.fit(data_pool[pool_indices])
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data_pool[pool_indices])
        imp_train_idx = [pool_indices[i] for i in closest]
        imp_cand_idx = [i for i in pool_indices if i not in imp_train_idx]
        init_method_str = "Smart Init (KMeans)"

    # 4. 设置迭代次数
    M = 300 if config.dataset_name == 'yeast' else 100
    print(f"实验设置: 迭代 {M} 轮")

    # 5. 运行实验
    print("\n>>> 开始运行原始算法...")
    acc_orig = run_experiment(
        original_algo, "Original (Baseline)", 
        data_pool, labels_pool, 
        orig_train_idx, orig_cand_idx, test_indices, 
        M, is_improved=False
    )

    print("\n>>> 开始运行改进算法...")
    acc_imp = run_experiment(
        improved_algo, f"Improved ({init_method_str})", 
        data_pool, labels_pool, 
        imp_train_idx, imp_cand_idx, test_indices, 
        M, is_improved=True
    )

    # 6. 创建结果对象
    result = ExperimentResult(config)
    result.original_acc = acc_orig
    result.improved_acc = acc_imp
    result.original_smooth = smooth_curve(acc_orig, config.smooth_weight)
    result.improved_smooth = smooth_curve(acc_imp, config.smooth_weight)
    result.init_method = init_method_str
    
    # 记录数据集统计信息
    result.dataset_stats = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': np.bincount(y).tolist()
    }
    
    result.runtime_info = {
        'M': M,
        'initial_train_size': 10,
        'candidate_size': len(pool_indices) - 10,
        'test_size': len(test_indices)
    }
    
    # 7. 保存结果
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    result_file = results_dir / f"{result.experiment_id}.json"
    result.save(result_file)
    
    # 同时保存为最新结果
    latest_file = results_dir / f"{config.dataset_name}_latest.json"
    result.save(latest_file)
    
    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"原始算法最终准确率: {acc_orig[-1]:.4f}")
    print(f"改进算法最终准确率: {acc_imp[-1]:.4f}")
    print(f"提升: {acc_imp[-1] - acc_orig[-1]:.4f}")
    print(f"结果文件: {result_file}")
    print(f"{'='*60}")
    
    # # 绘制简单图表
    # plot_simple_result(result)
    
    return result

# === 绘制简单结果图（可选）===
def plot_simple_result(result):
    """绘制简单的结果图"""
    plt.figure(figsize=(8, 5))
    
    iters = range(len(result.original_acc))
    
    plt.plot(iters, result.original_acc, color='gray', alpha=0.2, linewidth=1)
    plt.plot(iters, result.improved_acc, color='red', alpha=0.15, linewidth=1)
    plt.plot(iters, result.original_smooth, linestyle='--', color='gray', linewidth=2, label='Original D-TRUST')
    plt.plot(iters, result.improved_smooth, linestyle='-', color='#d62728', linewidth=2.5, label='Improved Method (Ours)')
    
    plt.title(f'Performance on {result.config["dataset_name"].capitalize()} Dataset')
    plt.xlabel('Number of Queries')
    plt.ylabel('Test Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 保存图表
    save_name = f'quick_plot_{result.config["dataset_name"]}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"快速图表已保存: {save_name}")

if __name__ == "__main__":
    # 运行实验
    result = run_full_experiment()
    
    if result:
        print(f"\n实验ID: {result.experiment_id}")
        print("要绘制更详细的图表，请运行: python plot_results.py")
    else:
        print("实验失败！")