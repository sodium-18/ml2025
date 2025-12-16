# -*- coding: utf-8 -*-
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# 导入你刚才保存的 sample.py 模块
import sample 

def main():
    # 1. 生成模拟数据 (模拟一个多分类问题)
    # 生成 500 个样本，10 个特征，3 个类别
    print("正在生成模拟数据...")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, 
                               n_classes=3, n_clusters_per_class=1, random_state=42)

    # 2. 划分数据集
    # 按照论文逻辑：初始训练集(L0)、候选池(U)、测试集(Test)
    # 这里我们切分：10% 测试，剩余的中取 20 个作为初始训练，剩下作为候选池
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 获取索引 (sample.py 需要索引列表)
    # 注意：sample.py 接收的是全局索引，但为了方便，我们这里简化逻辑，
    # 实际使用中，你需要维护全局数据的索引列表。
    
    # 重新组织数据以便通过索引访问
    data = np.vstack([X_train_full, X_test])
    labels = np.concatenate([y_train_full, y_test])
    
    all_indices = np.arange(len(data))
    test_start_idx = len(X_train_full)
    
    test_index = list(all_indices[test_start_idx:])
    train_pool_indices = all_indices[:test_start_idx]
    
    # 从训练池中随机选 20 个作为初始有标签数据
    np.random.shuffle(train_pool_indices)
    train_index = list(train_pool_indices[:20])      # 初始 L
    candidate_index = list(train_pool_indices[20:])  # 初始 U (候选池)

    print(f"数据准备完毕: 总数 {len(data)}, 初始训练 {len(train_index)}, 候选池 {len(candidate_index)}, 测试集 {len(test_index)}")

    # 3. 初始化 SVM 模型
    # 论文强调使用 OneVsRest (OVR) 和 RBF 核
    gam_clf = 0.1 # RBF 核参数 gamma
    c_clf = 10.0  # 惩罚系数 C (论文建议设大一点以区分 SV)
    
    clf_binary = svm.SVC(C=c_clf, probability=True, tol=0.001, kernel='rbf', gamma=gam_clf)
    clf = OneVsRestClassifier(clf_binary)

    # 4. 设置 D-TRUST 参数 (参考论文中的推荐值)
    M = 50          # 主动学习迭代次数 (想要跑得久可以改大，例如 100)
    GAM = 0         # gamma_0 (初始为0)
    GAM_UR = 0.002  # delta_gamma (增长率)
    LAM = 1.0       # lambda_0 (初始权重)
    LAM_UR = 0.01   # delta_lambda (衰减率)

    # # 这的准确率更高
    # M = 50          # 主动学习迭代次数 (想要跑得久可以改大，例如 100)
    # GAM = 0         # gamma_0 (初始为0)
    # GAM_UR = 0.004  # delta_gamma (增长率)
    # LAM = 1.5       # lambda_0 (初始权重)
    # LAM_UR = 0.005   # delta_lambda (衰减率)

    ##############################################################################################
    # prepare a mixed kernel function (alpha * RBF + (1-alpha) * linear)
    def mixed_kernel_factory(alpha=0.6, gamma=0.1):
        def kernel_fn(xA, xB):
            # xA, xB are 1xD numpy arrays
            r = rbf_kernel(np.vstack([xA, xB]), gamma=gamma)[0,1]
            l = float(np.dot(xA, xB.T))
            return alpha * r + (1.0 - alpha) * l
        return kernel_fn

    # 5. 实例化 sample 类
    # optional: use deep embedding as kernel input (deep-kernel)
    use_deep_kernel = False  # set True to enable PyTorch-based embedding

    # enable support-vector deduplication to avoid redundant samples in same region
    use_sv_dedup = True
    dedup_clusters = 20
    dedup_per_class = False

    print("初始化 D-TRUST 采样器...", f"use_sv_dedup={use_sv_dedup}", f"dedup_clusters={dedup_clusters}")
    # if deep kernel enabled, try to compute embeddings first
    if use_deep_kernel:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            class SimpleEncoder(nn.Module):
                def __init__(self, input_dim, emb_dim=32):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, emb_dim)
                    )
                def forward(self, x):
                    return self.net(x)

            # prepare small training for encoder using initial labeled set
            X_np = data.astype(np.float32)
            X_init = X_np[train_index]
            y_init = labels[train_index]
            enc = SimpleEncoder(X_np.shape[1], emb_dim=32)
            opt = optim.Adam(enc.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()
            enc.train()
            X_tensor = torch.from_numpy(X_init)
            y_tensor = torch.from_numpy(y_init.astype(np.longlong))
            # small number of epochs
            for ep in range(30):
                opt.zero_grad()
                out = enc(X_tensor)
                # small classifier head for self-supervised training (linear probe)
                # use linear classifier temporarily
                logits = out
                # if classes > embedding dim, still compute loss by projecting
                if logits.shape[1] < y_tensor.max()+1:
                    # project to num classes
                    proj = nn.Linear(logits.shape[1], int(y_tensor.max()+1))
                    logits = proj(out)
                loss = loss_fn(logits, y_tensor)
                loss.backward()
                opt.step()

            # compute embeddings for all data
            enc.eval()
            with torch.no_grad():
                all_emb = enc(torch.from_numpy(X_np)).numpy()
            data_to_use = all_emb
            print('使用 deep-embedding 作为 SVM 输入,shape=', data_to_use.shape)
        except Exception as e:
            print('无法启用 deep kernel(缺少 torch 或出错），回退到原始数据。错误：', e)
            data_to_use = data
    else:
        data_to_use = data

    dtrust_sampler = sample.sample(
        clf=clf,
        data=data_to_use,
        Y=labels,
        train_index=train_index,
        candidate_index=candidate_index,
        test_index=test_index,
        M=M,
        gam=GAM,
        gam_ur=GAM_UR,
        lam=LAM,
        lam_ur=LAM_UR,
        gam_clf=gam_clf,
        kernel_fn=mixed_kernel_factory(alpha=0.6, gamma=gam_clf),
        kernel='rbf',
        use_sv_dedup=use_sv_dedup,
        dedup_clusters=dedup_clusters,
        dedup_per_class=dedup_per_class
    )

    # 6. 开始主动学习循环
    print("开始主动学习循环...")
    accuracies = []
    
    # 先评估一下初始性能
    initial_acc = dtrust_sampler.evaluate()
    print(f"Iteration 0: Accuracy = {initial_acc:.4f}")
    accuracies.append(initial_acc)

    for i in range(M):
        # 执行采样 (dtrustSample 会选样本 ->以此更新训练集 -> 重训模型)
        # 这里的 n_batch=1 表示每次选 1 个样本
        dtrust_sampler.dtrustSample(n_batch=1)
        
        # 评估当前性能
        acc = dtrust_sampler.evaluate()
        accuracies.append(acc)
        
        print(f"Iteration {i+1}/{M}: Accuracy = {acc:.4f}, 训练集大小 = {len(dtrust_sampler.train_index)}")

    print("运行结束！")

if __name__ == "__main__":
    main()