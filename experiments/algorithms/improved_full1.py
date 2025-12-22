# -*- coding: utf-8 -*-
import scipy as sp
import scipy.stats
import numpy as np
import heapq 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans 
import operator

class sample(object):
    def __init__(self, clf, data, Y, train_index, candidate_index, test_index, M, gam_clf=0.1, **kwargs):
        self.clf = clf
        self.data = data
        self.Y = Y
        self.train_index = list(train_index)
        self.candidate_index = list(candidate_index)
        self.test_index = list(test_index)
        self.gam_clf = gam_clf 
        
        # 初始权重
        self.gam = 1.0
        self.lam = 1.0
        self.al_count = 0
        
        # === 【核心修改】确保接收动态参数 ===
        # 1. 衰减率 (Decay Rate)，默认 0.02
        self.gam_ur = kwargs.get('gam_ur', 0.02)
        
        # 2. Top-K 聚类数，默认 20
        self.dedup_clusters = kwargs.get('dedup_clusters', 20)
        self.use_sv_dedup = kwargs.get('use_sv_dedup', True)
        
        # 初始训练
        self.clf.fit(self.data[self.train_index, :], self.Y[self.train_index])
        
        # 计算初始对偶变量
        dualAve_list = []
        for i in range(len(self.clf.estimators_)):
            cur_E = self.clf.estimators_[i]
            if hasattr(cur_E, 'dual_coef_') and cur_E.dual_coef_ is not None:
                dualAve_list.append(np.mean(np.absolute(cur_E.dual_coef_)))
        self.dualAve_all_0 = np.mean(dualAve_list) if dualAve_list else 0.0
        self.prob = self.clf.predict_proba(self.data[self.candidate_index, :])

    def dtrustSample(self, n_batch=1):
        self.al_count += 1
        
        # 1. 计算 Evidence
        dec_Train = []
        sp_nList = []
        dualAve_list = []
        for i in range(len(self.clf.estimators_)):
            cur_E = self.clf.estimators_[i]
            dualCoef = cur_E.dual_coef_ if hasattr(cur_E, 'dual_coef_') else np.zeros((1,1))
            dualAve_list.append(np.mean(np.absolute(dualCoef)))
            sp_x = cur_E.support_vectors_
            sp_in = cur_E.support_
            sp_nList.append(len(sp_in))
            df_xlist = []
            Xgrid = self.data[self.candidate_index, :]
            for j in range(Xgrid.shape[0]):
                x_0 = Xgrid[j, :].reshape(1, -1)
                count = 0
                df_x0 = 0
                for sp_name in sp_in:
                    if count < dualCoef.shape[1] and (dualCoef[0, count] > 0):
                        x_spn = sp_x[count, :].reshape(1, -1)
                        xRBF = np.concatenate([x_0, x_spn], axis=0)
                        rbf = rbf_kernel(xRBF, gamma=self.gam_clf)[0, 1]
                        df_x0 += dualCoef[0, count] * rbf
                    count += 1
                df_xlist.append(df_x0)
            dec_Train.append(np.array(df_xlist).reshape(-1, 1))
        dec_Train = np.array(dec_Train).T.reshape(len(Xgrid), len(self.clf.estimators_))
        dualAve_all = np.mean(dualAve_list) if dualAve_list else 0
        
        # 2. 采样逻辑
        if dualAve_all < 1.5 * self.dualAve_all_0:
            mean_sp = np.mean(sp_nList) if np.mean(sp_nList) != 0 else 1.0
            resultList = np.amax(dec_Train / mean_sp, axis=1)
            
            # === 【修改点】使用 self.gam_ur ===
            decay_factor = np.exp(-self.gam_ur * self.al_count)
            current_lam = self.lam * decay_factor
            current_gam = self.gam * decay_factor
            
            final_scores = []
            for i in range(len(self.candidate_index)):
                probs = self.prob[i, :]
                dfpair = heapq.nlargest(2, probs)
                margin = (dfpair[0] - dfpair[1]) if len(dfpair) > 1 else 0
                ent = sp.stats.entropy(probs)
                score = (current_gam * resultList[i]) + (current_lam * margin) - ent
                final_scores.append(score)
            
            # === 【修改点】使用 self.dedup_clusters ===
            if self.use_sv_dedup:
                try:
                    n_cand = len(final_scores)
                    # 确保聚类数不超过候选样本数
                    n_clusters = min(self.dedup_clusters, n_cand)
                    
                    if n_clusters > 0 and n_clusters < n_cand: # 只有当需要聚类时才聚类
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(dec_Train)
                        labels = kmeans.labels_
                        cluster_reps = []
                        for cl in range(n_clusters):
                            members = [idx for idx, lab in enumerate(labels) if lab == cl]
                            if not members: continue
                            best_member = min(members, key=lambda idx: final_scores[idx])
                            cluster_reps.append(best_member)
                        cluster_reps = list(dict.fromkeys(cluster_reps))
                        targetIndexList = sorted(cluster_reps, key=lambda idx: final_scores[idx])[:n_batch]
                        if len(targetIndexList) < n_batch:
                            chosen = set(targetIndexList)
                            remaining = [i for i in range(len(final_scores)) if i not in chosen]
                            more = sorted(remaining, key=lambda idx: final_scores[idx])[:(n_batch - len(targetIndexList))]
                            targetIndexList.extend(more)
                    else:
                        targetIndexList = list(np.argsort(final_scores)[:n_batch])
                except:
                    targetIndexList = list(np.argsort(final_scores)[:n_batch])
            else:
                targetIndexList = list(np.argsort(final_scores)[:n_batch])
        else:
            ranking_scores = []
            for i in range(len(self.candidate_index)):
                ranking_scores.append(1.0 - np.max(dec_Train[i, :])) 
            targetIndexList = list(np.argsort(ranking_scores)[:n_batch])
            
        # 3. 更新
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