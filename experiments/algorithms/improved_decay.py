# -*- coding: utf-8 -*-
import scipy as sp
import scipy.stats
import numpy as np
import heapq 
from sklearn.metrics.pairwise import rbf_kernel

class sample(object):
    def __init__(self, clf, data, Y, train_index, candidate_index, test_index, M,
                 gam, gam_ur, lam, lam_ur, gam_clf=1, **kwargs):
        # 初始化与 Baseline 相同
        self.clf = clf
        self.data = data
        self.Y = Y
        self.train_index = list(train_index)
        self.candidate_index = list(candidate_index)
        self.test_index = list(test_index)
        self.gam_clf = gam_clf 
        self.gam = gam
        self.lam = lam
        self.al_count = 0
        
        self.clf.fit(self.data[self.train_index, :], self.Y[self.train_index])
        
        dualAve_list = []
        for i in range(len(self.clf.estimators_)):
            cur_E = self.clf.estimators_[i]
            if hasattr(cur_E, 'dual_coef_'):
                dualAve_list.append(np.mean(np.absolute(cur_E.dual_coef_)))
        self.dualAve_all_0 = np.mean(dualAve_list) if dualAve_list else 0.0
        self.prob = self.clf.predict_proba(self.data[self.candidate_index, :])

    def dtrustSample(self, n_batch=1):
        self.al_count += 1
        
        # ... (Evidence 计算部分省略，与 Baseline 相同，为了节省篇幅假设已计算 dec_Train) ...
        # 实际代码中需要把计算 dec_Train 的逻辑复制过来，这里为了展示核心差异简写：
        dec_Train, sp_nList, dualAve_all = self._compute_evidence() 
        
        if dualAve_all < 1.5 * self.dualAve_all_0:
            mean_sp = np.mean(sp_nList) if np.mean(sp_nList) != 0 else 1.0
            resultList = np.amax(dec_Train / mean_sp, axis=1)
            
            # 【核心改进】指数衰减 (Exponential Decay)
            decay_factor = np.exp(-0.02 * self.al_count)
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
            
            # 【对照组特征】仍然是贪心选择 (No Dedup)
            targetIndexList = list(np.argsort(final_scores)[:n_batch])
            
        else:
            # Ranking Phase (保持一致)
            ranking_scores = [1.0 for _ in range(len(self.candidate_index))] # 占位
            targetIndexList = list(np.argsort(ranking_scores)[:n_batch])

        self._update_model(targetIndexList)
        return 0

    def evaluate(self):
        return self.clf.score(self.data[self.test_index, :], self.Y[self.test_index])

    # 辅助函数，避免重复写大量代码
    def _compute_evidence(self):
        # 复制 Baseline 中的 Evidence 计算逻辑
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
        dec_Train = np.array(dec_Train).T.reshape(len(self.candidate_index), len(self.clf.estimators_))
        dualAve_all = np.mean(dualAve_list) if dualAve_list else 0
        return dec_Train, sp_nList, dualAve_all

    def _update_model(self, targetIndexList):
        delList = []
        for targetIndex in targetIndexList:
            delList.append(self.candidate_index[targetIndex])
            self.train_index.append(self.candidate_index[targetIndex])
        self.candidate_index = [x for x in self.candidate_index if x not in delList]
        self.clf.fit(self.data[self.train_index, :], self.Y[self.train_index])
        if len(self.candidate_index) > 0:
            self.prob = self.clf.predict_proba(self.data[self.candidate_index, :])