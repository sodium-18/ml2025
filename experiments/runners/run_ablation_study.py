# -*- coding: utf-8 -*-
import os
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

# ==========================================
# 导入我们的消融算法模块
# ==========================================
try:
    import baseline as baseline_algo
    print(">>> 成功导入: baseline (原始算法)")
except ImportError:
    print("错误：找不到 baseline.py")
    exit()

try:
    import improved_init as init_algo
    print(">>> 成功导入: improved_init (仅改进初始化)")
except ImportError:
    print("警告：找不到 improved_init.py")

try:
    import improved_decay as decay_algo
    print(">>> 成功导入: improved_decay (仅改进衰减)")
except ImportError:
    print("警告：找不到 improved_decay.py")

try:
    import improved_diversity as diversity_algo
    print(">>> 成功导入: improved_diversity (仅改进去重)")
except ImportError:
    print("警告：找不到 improved_diversity.py")

try:
    import improved_full as full_algo
    print(">>> 成功导入: improved_full (完整改进)")
except ImportError:
    print("警告：找不到 improved_full.py")

# ==========================================
# 数据加载函数
# ==========================================
def load_dataset(dataset_name):
    """加载UCI数据集"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if dataset_name == 'yeast':
        path = os.path.join(base_dir, "yeast.data")
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values
        
    elif dataset_name == 'dermatology':
        path = os.path.join(base_dir, "dermatology.data")
        df = pd.read_csv(path, header=None)
        df.replace('?', np.nan, inplace=True)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = df.iloc[:, :-1].values
        X = imputer.fit_transform(X)
        y = df.iloc[:, -1].values
        
    elif dataset_name == 'vehicle':
        path = os.path.join(base_dir, "vehicle.data")
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    # 标准化
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# ==========================================
# 实验运行函数
# ==========================================
def run_experiment(algo_module, algo_name, X, y, train_idx, cand_idx, test_idx, M, 
                   init_method='random', decay_method='linear', use_diversity=False):
    """运行单个算法配置"""
    print(f"\n>>> 运行: {algo_name}")
    print(f"    初始化: {init_method}, 衰减: {decay_method}, 去重: {use_diversity}")
    
    clf = OneVsRestClassifier(svm.SVC(C=10.0, probability=True, kernel='rbf', gamma=0.1))
    current_train = list(train_idx)
    current_cand = list(cand_idx)
    
    # 根据算法模块调用不同的初始化方式
    if 'init' in algo_name.lower() or 'full' in algo_name.lower():
        # 使用改进算法（需要额外参数）
        sampler = algo_module.sample(
            clf=clf, data=X, Y=y,
            train_index=current_train, candidate_index=current_cand, test_index=test_idx,
            M=M,
            gam=1.0, gam_ur=0.02, lam=1.0, lam_ur=0.02, gam_clf=0.1,
            use_sv_dedup=use_diversity,
            dedup_clusters=20
        )
    else:
        # 使用原始算法
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
        except Exception as e:
            print(f"  迭代 {i} 出错: {e}")
            break
            
        if (i+1) % 50 == 0:
            print(f"  迭代 {i+1}/{M}: 准确率 {acc:.4f}")
    
    return acc_list

# ==========================================
# K-Means初始化函数
# ==========================================
def kmeans_init(X, pool_indices, n_init=10):
    """使用K-Means选择初始训练样本"""
    kmeans = KMeans(n_clusters=n_init, random_state=42, n_init=10)
    kmeans.fit(X[pool_indices])
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X[pool_indices])
    train_idx = [pool_indices[i] for i in closest]
    cand_idx = [i for i in pool_indices if i not in train_idx]
    return train_idx, cand_idx

def random_init(pool_indices, n_init=10):
    """随机选择初始训练样本"""
    np.random.shuffle(pool_indices)
    train_idx = list(pool_indices[:n_init])
    cand_idx = list(pool_indices[n_init:])
    return train_idx, cand_idx

# ==========================================
# 主函数
# ==========================================
def main():
    # 1. 选择数据集
    DATASET_NAME = 'yeast'  # 可以改为 'dermatology', 'vehicle'
    print(f"=== 在 {DATASET_NAME} 数据集上运行消融实验 ===")
    
    # 2. 加载数据
    X, y = load_dataset(DATASET_NAME)
    
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
    
    # 4. 定义消融实验组
    experiment_groups = [
        # (算法模块, 算法名称, 初始化方法, 衰减方法, 是否去重)
        ('baseline', 'Baseline (原始)', 'random', 'linear', False),
        ('improved_init', '仅改进初始化', 'kmeans', 'linear', False),
        ('improved_decay', '仅改进衰减', 'random', 'exponential', False),
        ('improved_diversity', '仅改进去重', 'random', 'linear', True),
        ('improved_full', '完整改进', 'kmeans', 'exponential', True)
    ]
    
    # 5. 运行所有实验组
    results = {}
    M = 150  # 迭代次数
    
    for group in experiment_groups:
        algo_key, algo_name, init_method, decay_method, use_diversity = group
        
        # 准备初始训练集
        if init_method == 'kmeans':
            train_idx, cand_idx = kmeans_init(data_pool, pool_indices, n_init=10)
        else:
            np.random.seed(42)  # 固定种子
            train_idx, cand_idx = random_init(pool_indices.copy(), n_init=10)
        
        # 导入对应的算法模块
        if algo_key == 'baseline':
            algo_module = baseline_algo
        elif algo_key == 'improved_init':
            algo_module = init_algo
        elif algo_key == 'improved_decay':
            algo_module = decay_algo
        elif algo_key == 'improved_diversity':
            algo_module = diversity_algo
        elif algo_key == 'improved_full':
            algo_module = full_algo
        else:
            continue
        
        # 运行实验
        acc_history = run_experiment(
            algo_module, algo_name, data_pool, labels_pool,
            train_idx, cand_idx, test_indices, M,
            init_method, decay_method, use_diversity
        )
        
        results[algo_name] = acc_history
        print(f"✓ {algo_name} 完成, 最终准确率: {acc_history[-1]:.4f}")
    
    # 6. 绘制结果
    plot_results(results, DATASET_NAME)

def plot_results(results, dataset_name):
    """绘制消融实验对比图"""
    plt.figure(figsize=(12, 8))
    
    # 颜色和线型
    colors = {
        'Baseline (原始)': 'gray',
        '仅改进初始化': 'blue',
        '仅改进衰减': 'green',
        '仅改进去重': 'orange',
        '完整改进': 'red'
    }
    
    line_styles = {
        'Baseline (原始)': '--',
        '仅改进初始化': ':',
        '仅改进衰减': '-.',
        '仅改进去重': ':',
        '完整改进': '-'
    }
    
    # 绘制每条曲线
    for algo_name, acc_history in results.items():
        if algo_name in colors:
            x = range(len(acc_history))
            plt.plot(x, acc_history, 
                    color=colors[algo_name], 
                    linestyle=line_styles[algo_name],
                    linewidth=2.5 if algo_name == '完整改进' else 1.5,
                    label=algo_name,
                    alpha=0.9 if algo_name == '完整改进' else 0.7)
    
    # 添加阴影区域表示95%置信区间（如果有多次运行）
    plt.fill_between([0, len(list(results.values())[0])-1], 
                     [0.5, 0.5], [1.0, 1.0], 
                     alpha=0.05, color='gray')
    
    plt.title(f'Ablation Study on {dataset_name.capitalize()} Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Queries', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12, loc='lower right')
    plt.ylim(0.4, 1.0)
    
    # 添加改进贡献度标注
    if '完整改进' in results and 'Baseline (原始)' in results:
        final_improvement = results['完整改进'][-1] - results['Baseline (原始)'][-1]
        plt.text(0.02, 0.98, f'最终改进: +{final_improvement:.3f}', 
                transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存图片
    save_name = f'ablation_study_{dataset_name}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 消融实验图已保存为: {save_name}")
    
    # 显示表格数据
    print("\n=== 最终准确率对比 ===")
    print("{:<20} {:<10} {:<10}".format("算法", "最终准确率", "相对改进"))
    baseline_acc = results['Baseline (原始)'][-1] if 'Baseline (原始)' in results else 0
    
    for algo_name, acc_history in results.items():
        final_acc = acc_history[-1]
        improvement = final_acc - baseline_acc
        print("{:<20} {:<10.4f} {:<10.4f}".format(algo_name, final_acc, improvement))
    
    plt.show()

if __name__ == "__main__":
    main()