# -*- coding: utf-8 -*-
# plot_results.py - 专门用于绘图（保持原始风格）
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# === 配置 ===
DEFAULT_DATASET = 'vehicle'
RESULTS_DIR = "experiment_results"

# === 平滑函数（与run_experiment.py保持一致）===
def smooth_curve(points, weight=0.8):
    """指数移动平均平滑"""
    last = points[0]
    smoothed = []
    for point in points:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# === 加载结果 ===
def load_results(dataset_name=None, experiment_id=None):
    """加载实验结果"""
    results_dir = Path(RESULTS_DIR)
    
    if experiment_id:
        # 加载指定实验ID的结果
        result_file = results_dir / f"{experiment_id}.json"
    elif dataset_name:
        # 加载指定数据集的最新结果
        result_file = results_dir / f"{dataset_name}_latest.json"
    else:
        # 列出所有可用结果
        print("可用结果文件:")
        json_files = list(results_dir.glob("*.json"))
        for i, file in enumerate(json_files, 1):
            print(f"{i}. {file.name}")
        
        choice = input("\n请选择文件编号或输入文件名: ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(json_files):
                result_file = json_files[choice_idx]
            else:
                print("编号无效")
                return None
        except ValueError:
            result_file = results_dir / choice
    
    if not result_file.exists():
        print(f"结果文件不存在: {result_file}")
        return None
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        print(f"加载结果: {data['experiment_id']}")
        print(f"数据集: {data['dataset']}")
        print(f"时间: {data['timestamp']}")
        print(f"迭代轮数: {len(data['original_acc'])-1}")
        return data
    except Exception as e:
        print(f"加载结果失败: {e}")
        return None

# === 主绘图函数（保持原始风格）===
def plot_original_style(result_data, save_figure=True, show_figure=True, 
                        smooth_weight=0.8, ylim_adjust=True):
    """绘制原始风格的对比图（Queries改为Rounds）"""
    dataset_name = result_data['dataset']
    
    # 获取数据
    acc_orig = result_data['original_acc']
    acc_imp = result_data['improved_acc']
    
    # 重新计算平滑曲线（如果需要不同的平滑权重）
    if 'original_smooth' in result_data and smooth_weight == 0.8:
        smooth_orig = result_data['original_smooth']
        smooth_imp = result_data['improved_smooth']
    else:
        smooth_orig = smooth_curve(acc_orig, weight=smooth_weight)
        smooth_imp = smooth_curve(acc_imp, weight=smooth_weight)
    
    # 创建图形（保持原始大小）
    plt.figure(figsize=(10, 6))
    
    # 确定迭代轮数（处理不同数据集的轮次差异）
    total_rounds = len(acc_orig) - 1  # 减去初始轮
    iters = range(len(acc_orig))      # 包括初始轮（round 0）
    
    print(f"绘图信息:")
    print(f"  - 总轮次: {total_rounds}")
    print(f"  - 数据点数量: {len(acc_orig)}")
    print(f"  - 平滑权重: {smooth_weight}")
    
    # 绘制阴影（真实波动）- 保持原始透明度
    plt.plot(iters, acc_orig, color='gray', alpha=0.2, linewidth=1)
    plt.plot(iters, acc_imp, color='red', alpha=0.15, linewidth=1)
    
    # 绘制平滑曲线（主视觉）- 保持原始样式
    plt.plot(iters, smooth_orig, linestyle='--', color='gray', linewidth=2, 
             label='Original D-TRUST')
    plt.plot(iters, smooth_imp, linestyle='-', color='#d62728', linewidth=2.5, 
             label='Improved Method (Ours)')
    
    # 设置标题和标签 - Queries改为Rounds
    plt.title(f'Performance Comparison on {dataset_name.capitalize()} Dataset', 
              fontsize=15, fontweight='bold')
    plt.xlabel('Number of Rounds', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    
    # 设置x轴刻度 - 根据轮次数量调整
    if total_rounds <= 100:
        x_ticks = list(range(0, len(iters), max(1, len(iters)//10)))
        plt.xticks(x_ticks)
    else:
        # 对于较多轮次，使用更稀疏的刻度
        x_ticks = list(range(0, len(iters), max(1, len(iters)//20)))
        plt.xticks(x_ticks)
    
    # 设置y轴范围 - 保持原始逻辑但更灵活
    if ylim_adjust:
        # 获取y值范围
        all_acc = acc_orig + acc_imp
        y_min = min(all_acc)
        y_max = max(all_acc)
        
        # 根据不同数据集调整y轴范围
        if dataset_name == 'dermatology':
            plt.ylim(max(0.5, y_min * 0.95), min(1.0, y_max * 1.05))
        elif dataset_name == 'yeast':
            plt.ylim(max(0.4, y_min * 0.95), min(0.65, y_max * 1.05))
        else:
            # 其他数据集：留出10%的边距
            y_range = y_max - y_min
            if y_range > 0:
                plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            else:
                plt.ylim(y_min * 0.95, y_max * 1.05)
    else:
        # 不调整y轴范围
        pass
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 图例位置根据最终准确率调整
    if acc_imp[-1] > acc_orig[-1]:
        legend_loc = 'lower right'
    else:
        legend_loc = 'upper right'
    
    plt.legend(fontsize=12, loc=legend_loc)
    
    # 在右上角添加最终准确率信息
    final_text = f'Final: Original={acc_orig[-1]:.4f}, Improved={acc_imp[-1]:.4f}'
    plt.annotate(final_text, xy=(0.98, 0.02), xycoords='axes fraction',
                 fontsize=11, ha='right', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 在左上角添加数据集和轮次信息
    info_text = f'{dataset_name.upper()} Dataset\n{total_rounds} Rounds'
    plt.annotate(info_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_figure:
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f'compare_{dataset_name}_{total_rounds}rounds_{timestamp}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"\n对比图已保存为: {save_name}")
    
    if show_figure:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()

# === 批量绘图函数 ===
def plot_all_datasets(smooth_weight=0.8, show_figure=True):
    """绘制所有可用数据集的图表"""
    results_dir = Path(RESULTS_DIR)
    
    # 查找所有数据集的最新结果
    datasets = ['dermatology', 'yeast', 'vehicle', 'segment']
    available_datasets = []
    
    for dataset in datasets:
        latest_file = results_dir / f"{dataset}_latest.json"
        if latest_file.exists():
            available_datasets.append(dataset)
    
    if not available_datasets:
        print("未找到任何实验结果，请先运行实验。")
        return
    
    print(f"找到 {len(available_datasets)} 个数据集的实验结果:")
    for dataset in available_datasets:
        print(f"  - {dataset}")
    
    # 绘制每个数据集的图表
    for dataset in available_datasets:
        result_data = load_results(dataset)
        if result_data:
            print(f"\n正在绘制 {dataset} 的图表...")
            plot_original_style(result_data, save_figure=True, 
                               show_figure=show_figure, smooth_weight=smooth_weight)
    
    print(f"\n所有图表绘制完成！")

# === 创建子图对比 ===
def plot_subplot_comparison(datasets=None, figsize=(15, 10)):
    """创建多个数据集的子图对比"""
    results_dir = Path(RESULTS_DIR)
    
    if datasets is None:
        # 默认绘制所有可用的数据集
        datasets = ['dermatology', 'yeast', 'vehicle', 'segment']
    
    # 过滤可用的数据集
    available_datasets = []
    for dataset in datasets:
        latest_file = results_dir / f"{dataset}_latest.json"
        if latest_file.exists():
            available_datasets.append(dataset)
        else:
            print(f"警告: 未找到 {dataset} 的结果文件")
    
    if not available_datasets:
        print("未找到任何实验结果")
        return
    
    # 计算子图布局
    n_plots = len(available_datasets)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, dataset in enumerate(available_datasets):
        ax = axes[i]
        
        # 加载结果
        result_file = results_dir / f"{dataset}_latest.json"
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        # 获取数据
        acc_orig = result_data['original_acc']
        acc_imp = result_data['improved_acc']
        smooth_orig = result_data['original_smooth']
        smooth_imp = result_data['improved_smooth']
        
        iters = range(len(acc_orig))
        
        # 在子图中绘制
        ax.plot(iters, acc_orig, color='gray', alpha=0.2, linewidth=1)
        ax.plot(iters, acc_imp, color='red', alpha=0.15, linewidth=1)
        ax.plot(iters, smooth_orig, linestyle='--', color='gray', linewidth=2, 
                label='Original')
        ax.plot(iters, smooth_imp, linestyle='-', color='#d62728', linewidth=2.5, 
                label='Improved')
        
        ax.set_title(f'{dataset.upper()} Dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Rounds')
        ax.set_ylabel('Test Accuracy')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        
        # 添加最终准确率信息
        final_text = f'Final: {acc_imp[-1]:.4f} vs {acc_orig[-1]:.4f}'
        ax.annotate(final_text, xy=(0.98, 0.02), xycoords='axes fraction',
                    fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Active Learning Performance Comparison Across Datasets', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f'multi_dataset_comparison_{timestamp}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"多数据集对比图已保存为: {save_name}")
    
    plt.show()
    return fig

# === 主函数 ===
def main():
    parser = argparse.ArgumentParser(description='绘制实验结果图（原始风格）')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, 
                       help=f'数据集名称 (默认: {DEFAULT_DATASET})')
    parser.add_argument('--experiment', type=str, 
                       help='实验ID (例如: vehicle_20240101_120000)')
    parser.add_argument('--smooth', type=float, default=0.8,
                       help='平滑权重 (默认: 0.8)')
    parser.add_argument('--no-adjust-ylim', action='store_true',
                       help='不自动调整y轴范围')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用结果')
    parser.add_argument('--all', action='store_true',
                       help='绘制所有可用数据集的图表')
    parser.add_argument('--subplot', action='store_true',
                       help='创建多个数据集的子图对比')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图表，仅保存文件')
    
    args = parser.parse_args()
    
    if args.list:
        # 列出所有结果
        results_dir = Path(RESULTS_DIR)
        json_files = list(results_dir.glob("*.json"))
        print(f"\n可用结果文件 ({len(json_files)} 个):")
        
        # 按类型分组
        latest_files = [f for f in json_files if f.name.endswith('_latest.json')]
        history_files = [f for f in json_files if not f.name.endswith('_latest.json')]
        
        if latest_files:
            print("\n最新结果:")
            for file in latest_files:
                dataset = file.name.replace('_latest.json', '')
                print(f"  {dataset:15s} -> {file.name}")
        
        if history_files:
            print("\n历史结果:")
            for file in sorted(history_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                print(f"  {file.name:30s} ({file_time})")
        
        if len(history_files) > 10:
            print(f"  ... 还有 {len(history_files)-10} 个历史结果")
        
        return
    
    if args.all:
        # 绘制所有数据集的图表
        plot_all_datasets(smooth_weight=args.smooth, show_figure=not args.no_show)
        return
    
    if args.subplot:
        # 创建子图对比
        plot_subplot_comparison()
        return
    
    # 加载单个结果
    result_data = load_results(args.dataset if not args.experiment else None, args.experiment)
    if not result_data:
        print(f"未找到结果文件，请先运行 run_experiment.py")
        print(f"或者使用 --list 参数查看可用结果")
        return
    
    # 绘制图表
    print(f"\n正在绘制图表...")
    plot_original_style(result_data, 
                       save_figure=True, 
                       show_figure=not args.no_show,
                       smooth_weight=args.smooth,
                       ylim_adjust=not args.no_adjust_ylim)
    
    print(f"\n图表信息:")
    print(f"  数据集: {result_data['dataset']}")
    print(f"  实验ID: {result_data['experiment_id']}")
    print(f"  总轮次: {len(result_data['original_acc'])-1}")
    print(f"  最终准确率 - 原始: {result_data['original_acc'][-1]:.4f}")
    print(f"  最终准确率 - 改进: {result_data['improved_acc'][-1]:.4f}")
    print(f"  提升幅度: {result_data['improved_acc'][-1] - result_data['original_acc'][-1]:.4f}")

if __name__ == "__main__":
    main()