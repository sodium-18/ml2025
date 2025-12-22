# -*- coding: utf-8 -*-
import scipy as sp
import scipy.stats
import numpy as np
import heapq 
from sklearn.metrics.pairwise import rbf_kernel
import operator

def dissonance(b):
    def Bal(b,i,j):
        return 1-np.abs(b[i]-b[j])/(b[i]+b[j])
    res=0
    for i in range(len(b)):
        excludeInd=[x for x in range(len(b)) if x != i]
        tem1=0
        tem2=0
        for j in excludeInd:
            tem1+=(b[j]*Bal(b,i,j))
            tem2+=(b[j])
        if(tem2==0):
            return 0
        res+=b[i]*tem1/tem2
    return res

class sample(object):
    def __init__(self, clf, data, Y, train_index, candidate_index, test_index, M,
                 gam, gam_ur, lam, lam_ur, gam_clf=1):
        self.clf = clf
        self.data = data
        self.Y = Y
        self.train_index=list(train_index)
        self.candidate_index=list(candidate_index)
        self.test_index=list(test_index)
        
        self.clf.fit(self.data[self.train_index,:], self.Y[self.train_index])
        
        dualAve_list = []
        for i in range(len(self.clf.estimators_)):
            cur_E = self.clf.estimators_[i]
            dualCoef = cur_E.dual_coef_
            dualAve = np.mean(np.absolute(dualCoef))
            dualAve_list.append(dualAve)
        self.dualAve_all_0 = np.mean(dualAve_list) if len(dualAve_list)>0 else 0
        
        self.prob = self.clf.predict_proba(self.data[self.candidate_index,:])
        self.al_count = 0
        self.M = M
        self.gam = gam
        self.lam = lam
        self.gam_ur = gam_ur
        self.lam_ur = lam_ur
        self.currentPerformance=0
        self.gam_clf = gam_clf
   
    def dtrustSample(self, n_batch=1):
        self.al_count += 1
        dec_Train=[]
        sp_nList = []
        dualAve_list = []
        
        for i in range(len(self.clf.estimators_)):
            cur_E = self.clf.estimators_[i]
            dualCoef = cur_E.dual_coef_
            dualAve = np.mean(np.absolute(dualCoef))
            dualAve_list.append(dualAve)
            sp_in = cur_E.support_
            sp_nList.append(len(sp_in))
            sp_x = cur_E.support_vectors_
            df_xlist = []
            Xgrid = self.data[self.candidate_index,:]
            
            for j in range(Xgrid.shape[0]):
                x_0 = Xgrid[j,:].reshape(1,-1)
                count = 0
                df_x0 = 0
                for sp_name in sp_in:
                    if count < dualCoef.shape[1] and dualCoef[0,count]>0:
                        x_spn = sp_x[count,:].reshape(1,-1)
                        xRBF = np.concatenate([x_0,x_spn], axis=0)
                        rbf = rbf_kernel(xRBF, gamma=self.gam_clf)[0,1]
                        df_x0 = df_x0 + dualCoef[0,count]*rbf
                    count += 1
                df_xlist.append(df_x0)
            dec_Train.append(np.array(df_xlist).reshape(-1,1))
            
        dec_Train = np.array(dec_Train).T.reshape(len(Xgrid), len(self.clf.estimators_))
        dualAve_all = np.mean(dualAve_list) if len(dualAve_list)>0 else 0
        
        if dualAve_all < 1.5*self.dualAve_all_0:
            # 线性衰减（原始方法）
            resultList = np.amax(dec_Train/np.mean(sp_nList), axis=1).tolist()
            
            dfpairList = []
            lam_weight = max(0.0, self.lam - self.al_count*self.lam_ur)  # 线性衰减
            for i in range(len(self.candidate_index)):
                dfpair = heapq.nlargest(2, self.prob[i,:])
                margin = (dfpair[0] - dfpair[1]) if len(dfpair)>1 else 0
                dfpairList.append(lam_weight*margin)
            
            dfallList = []
            for i in range(len(self.candidate_index)):
                ent = sp.stats.entropy(self.prob[i,:])
                if self.al_count > 0.1*self.M:
                    dfallList.append(ent)
                else:
                    dfallList.append(0)
            
            mcList = list(map(operator.sub, dfpairList, dfallList))
            dfList = []
            for i in range(len(mcList)):
                if self.al_count > 0.1*self.M:
                    gam_weight = max(0.0, self.gam - self.al_count*self.gam_ur)  # 线性衰减
                    dfGam = gam_weight*resultList[i]
                    if dfGam > 0:
                        dfList.append(dfGam)
                    else:
                        dfList.append(0)
                else:
                    dfList.append(self.al_count*self.gam_ur*resultList[i])
            
            resultList = list(map(operator.add, dfList, mcList))
            targetIndexList = list(np.argsort(resultList)[:n_batch])
            
        else:
            resultList = []
            for i in range(len(Xgrid)):
                dis = dissonance(dec_Train[i,:])
                resultList.append(dis)
            targetIndexList = list(np.argsort(resultList)[-n_batch:])
        
        delList = []
        for targetIndex in targetIndexList:
            delList.append(self.candidate_index[targetIndex])
            self.train_index.append(self.candidate_index[targetIndex])
        self.candidate_index = [x for x in self.candidate_index if x not in delList]
        
        self.clf.fit(self.data[self.train_index,:], self.Y[self.train_index])
        self.prob = self.clf.predict_proba(self.data[self.candidate_index,:])
        return 0
    
    def evaluate(self):
        self.currentPerformance = self.clf.score(self.data[self.test_index,:], self.Y[self.test_index])
        return self.currentPerformance