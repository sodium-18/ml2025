# 在baseline.py基础上，修改__init__来接受K-Means初始化
# 但采样逻辑保持原始线性衰减
class sample(object):
    def __init__(self, clf, data, Y, train_index, candidate_index, test_index, M,
                 gam, gam_ur, lam, lam_ur, gam_clf=1, use_sv_dedup=False, dedup_clusters=10):
        # ... 与baseline相同的初始化 ...
        # 但添加了use_sv_dedup参数（虽然不用，但保持接口一致）
        pass
    
    def dtrustSample(self, n_batch=1):
        # 完全使用baseline的线性衰减逻辑
        # 不启用去重（即使参数设置了）
        pass