# -*- coding: utf-8 -*-
import scipy as sp
import scipy.stats
import numpy as np
import heapq 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans # 必须引入这个用于去重
import operator

# === 辅助函数：计算不一致性 (直接包含在这里，不用去别处找) ===
def dissonance(b):
    def Bal(b,i,j):
        denom = b[i] + b[j]
        if denom < 1e-9: return 0
        return 1 - np.abs(b[i] - b[j]) / denom
    
    res = 0
    for i in range(len(b)):
        excludeInd = [x for x in range(len(b)) if x != i]
        tem1 = 0
        tem2 = 0
        for j in excludeInd:
            tem1 += (b[j] * Bal(b, i, j))
            tem2 += (b[j])
        
        if tem2 < 1e-9:
            return 0
        res += b[i] * tem1 / tem2
    return res

class sample(object):
    def __init__(self, clf, data, Y, train_index, candidate_index, test_index, M,
                 gam, gam_ur, lam, lam_ur, gam_clf=1, use_sv_dedup=True, dedup_clusters=20, **kwargs):
        self.clf = clf
        self.data = data
        self.Y = Y
        self.train_index = list(train_index)
        self.candidate_index = list(candidate_index)
        self.test_index = list(test_index)
        self.gam_clf = gam_clf 
        
        # 初始参数 (线性衰减需要用到 gam_ur 和 lam_ur)
        self.gam = gam
        self.lam = lam
        self.gam_ur = gam_ur
        self.lam_ur = lam_ur
        self.al_count = 0
        
        # 核心消融参数：是否开启去重
        self.use_sv_dedup = use_sv_dedup
        self.dedup_clusters = dedup_clusters
        
        # 初始训练
        self.clf.fit(self.data[self.train_index, :], self.Y[self.train_index])
        
        # 计算初始对偶变量 (用于判断是否切换阶段)
        dualAve_list = []
        for i in range(len(self.clf.estimators_)):
            cur_E = self.clf.estimators_[i]
            if hasattr(cur_E, 'dual_coef_') and cur_E.dual_coef_ is not None:
                dualAve = np.mean(np.absolute(cur_E.dual_coef_))
                dualAve_list.append(dualAve)
        
        if len(dualAve_list) > 0:
            self.dualAve_all_0 = np.mean(dualAve_list)
        else:
            self.dualAve_all_0 = 0.0

        self.prob = self.clf.predict_proba(self.data[self.candidate_index, :])

    def dtrustSample(self, n_batch=1):
        self.al_count += 1
        
        # 1. 计算 Evidence (从 baseline 完整复制过来的逻辑)
        dec_Train = []
        sp_nList = []
        dualAve_list = []
        
        for i in range(len(self.clf.estimators_)):
            cur_E = self.clf.estimators_[i]
            dualCoef = cur_E.dual_coef_ if hasattr(cur_E, 'dual_coef_') else np.zeros((1,1))
            dualAve = np.mean(np.absolute(dualCoef))
            dualAve_list.append(dualAve)
            
            sp_x = cur_E.support_vectors_
            sp_in = cur_E.support_
            sp_nList.append(len(sp_in))
            
            df_xlist = []
            Xgrid = self.data[self.candidate_index, :]
            
            # 计算 RBF 核距离
            for j in range(Xgrid.shape[0]):
                x_0 = Xgrid[j, :].reshape(1, -1)
                count = 0
                df_x0 = 0
                for sp_name in sp_in:
                    if count < dualCoef.shape[1] and (dualCoef[0, count] > 0):
                        x_spn = sp_x[count, :].reshape(1, -1)
                        xRBF = np.concatenate([x_0, x_spn], axis=0)
                        rbf = rbf_kernel(xRBF, gamma=self.gam_clf)[0, 1]
                        df_x0 = df_x0 + dualCoef[0, count] * rbf
                    count = count + 1
                df_xlist.append(df_x0)
            dec_Train.append(np.array(df_xlist).reshape(-1, 1))
            
        dec_Train = np.array(dec_Train).T.reshape(len(Xgrid), len(self.clf.estimators_))
        dualAve_all = np.mean(dualAve_list) if len(dualAve_list) > 0 else 0
        
        # 2. 探索 vs 排序
        if dualAve_all < 1.5 * self.dualAve_all_0:
            # === Discovery Phase ===
            mean_sp = np.mean(sp_nList)
            if mean_sp == 0: mean_sp = 1.0
            
            resultList = np.amax(dec_Train / mean_sp, axis=1)
            
            # 【对照组特征】保持线性衰减 (Linear Decay)
            # 只有这里和 sample_final.py 不同 (那里是指数衰减)
            current_lam = max(0.0, self.lam - self.al_count * self.lam_ur)
            current_gam = max(0.0, self.gam - self.al_count * self.gam_ur)
            
            final_scores = []
            for i in range(len(self.candidate_index)):
                probs = self.prob[i, :]
                dfpair = heapq.nlargest(2, probs)
                margin = (dfpair[0] - dfpair[1]) if len(dfpair) > 1 else 0
                ent = sp.stats.entropy(probs)
                
                # 线性组合计算分数
                score = (current_gam * resultList[i]) + (current_lam * margin) - ent
                final_scores.append(score)
            
            # 【核心改进】Top-K 聚类去重
            # 这是这个文件的核心：用线性衰减，但开启去重
            if self.use_sv_dedup and len(final_scores) > 0:
                try:
                    n_cand = len(final_scores)
                    n_clusters = min(self.dedup_clusters, n_cand)
                    
                    if n_clusters > 0:
                        # 对 Evidence 空间进行聚类
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(dec_Train)
                        labels = kmeans.labels_
                        
                        cluster_reps = []
                        for cl in range(n_clusters):
                            # 找到该簇的所有成员
                            members = [idx for idx, lab in enumerate(labels) if lab == cl]
                            if not members: continue
                            # 在簇内选分数最小(最好)的那个
                            best_member = min(members, key=lambda idx: final_scores[idx])
                            cluster_reps.append(best_member)
                        
                        cluster_reps = list(dict.fromkeys(cluster_reps))
                        # 从代表中选最好的 n_batch 个
                        targetIndexList = sorted(cluster_reps, key=lambda idx: final_scores[idx])[:n_batch]
                        
                        # 如果不够 n_batch 个，从剩下的补
                        if len(targetIndexList) < n_batch:
                            chosen = set(targetIndexList)
                            remaining = [i for i in range(len(final_scores)) if i not in chosen]
                            more = sorted(remaining, key=lambda idx: final_scores[idx])[:(n_batch - len(targetIndexList))]
                            targetIndexList.extend(more)
                    else:
                        targetIndexList = list(np.argsort(final_scores)[:n_batch])
                except:
                    # 如果聚类失败，回退到普通排序
                    targetIndexList = list(np.argsort(final_scores)[:n_batch])
            else:
                targetIndexList = list(np.argsort(final_scores)[:n_batch])

        else:
            # === Ranking Phase (保持一致) ===
            ranking_scores = []
            for i in range(len(Xgrid)):
                dis = dissonance(dec_Train[i, :])
                ranking_scores.append(dis)
            targetIndexList = list(np.argsort(ranking_scores)[-n_batch:])
            
        # 3. 更新模型
        delList = []
        for targetIndex in targetIndexList:
            delList.append(self.candidate_index[targetIndex])
            self.train_index.append(self.candidate_index[targetIndex])
        self.candidate_index = [x for x in self.candidate_index if x not in delList]
        
        self.clf.fit(self.data[self.train_index, :], self.Y[self.train_index])
        if len(self.candidate_index) > 0:
            self.prob = self.clf.predict_proba(self.data[self.candidate_index, :])
        return 0

    def evaluate(self):
        return self.clf.score(self.data[self.test_index, :], self.Y[self.test_index])