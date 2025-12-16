# -*- coding: utf-8 -*-
import scipy as sp
import scipy.stats # 新增这一行
import numpy as np
import heapq 
from sklearn.metrics.pairwise import rbf_kernel
import operator

def dissonance(b):
        #def Bal(i,j)
        def Bal(b,i,j):
            return 1-np.abs(b[i]-b[j])/(b[i]+b[j])
        #get evidence
        res=0
        for i in range(len(b)):
            excludeInd=[x for x in range(len(b)) if x != i]
            tem1=0
            tem2=0
            for j in excludeInd:
                tem1+=(b[j]*Bal(b,i,j))
                tem2+=(b[j])
            if(tem2==0):
                #print(0)
                return 0
            res+=b[i]*tem1/tem2
        #print(res)
        return res


class sample( object ):
#    initialization
    def __init__( self, clf, data, Y, \
                 train_index, candidate_index, test_index, M,\
                 gam, gam_ur, lam, lam_ur, gam_clf = 1 ):
#        basic initializations
#        classifier
        self.clf = clf
#        data(x)
        self.data = data
#        label(t)
        self.Y = Y
#        training, candidate, and testing index
        self.train_index=list(train_index)
        self.candidate_index=list(candidate_index)
        self.test_index=list(test_index)
#        initialize the model
        self.clf.fit( self.data[ self.train_index , : ] , \
                     self.Y[ self.train_index] )
#        compute the average dual variable value        
        for i in range(len(self.clf.estimators_)):
#            this is using OneVsRestClassifier()
            cur_E = self.clf.estimators_[i]
            dualCoef = cur_E.dual_coef_
            dualAve = np.mean(np.absolute(dualCoef))
        self.dualAve_all_0 = np.mean(dualAve)# <--- 加上 self.
#        get the predictive probabilities over candidates
        self.prob = self.clf.predict_proba( \
                            self.data[ self.candidate_index , : ] )
        self.nc = len( set(Y) )

#        iteration count
        self.al_count = 0
        self.M = M
#        initial balance parameter
        self.gam = gam
        self.lam = lam
#        Update rate
        self.gam_ur = gam_ur

        self.lam_ur = lam_ur
#        evaluation
        self.currentPerformance=0
#        gam_clf for rbf kernel (=1/2l^2), l = length scale
        self.gam_clf = gam_clf
   
    def dtrustSample(self,n_batch = 1):
        self.al_count = self.al_count + 1
#        compute S_DF
        dec_Train=[]
        sp_nList = []
        dual_List = []
        dual_sampled = []
        # compute D_t for each class
        for i in range(len(self.clf.estimators_)):
#            this is using OneVsRestClassifier()
            cur_E = self.clf.estimators_[i]
            dualCoef = cur_E.dual_coef_
            dualAve = np.mean(np.absolute(dualCoef))
            dual_p = dualCoef[dualCoef>0]
            dual_sampled.append(dual_p[-1])
            dual_List.append(dualCoef)
            sp_x = cur_E.support_vectors_
            sp_in = cur_E.support_
            sp_nList.append(len(sp_in))
            df_xlist = []
            Xgrid = self.data[self.candidate_index,:]
            for j in range(Xgrid.shape[0]):
                x_0 = Xgrid[j,:].reshape(1,-1)
                count = 0
                df_x0 = 0
                for sp_name in sp_in:
                    if (dualCoef[0,count]>0):
                        
                        x_spn = sp_x[count,:].reshape(1,-1)
                        xRBF = np.concatenate([x_0,x_spn],axis = 0)
                        rbf = rbf_kernel(xRBF, gamma = self.gam_clf)[0,1]
                        df_x0 = df_x0 + dualCoef[0,count]*rbf
                    count = count+1 
                df_xlist.append(df_x0)
            dec_Train.append(np.array(df_xlist).reshape(-1,1))
        dec_Train = np.array(dec_Train).T.reshape(len(Xgrid), \
                             len(self.clf.estimators_))
        dualAve_all = np.mean(dualAve)                     
        #compute the sampling score based on al_count and dualAve
        if dualAve_all<1.5*self.dualAve_all_0:# <--- 加上 self.
        
            resultList = np.amax(dec_Train/np.mean(sp_nList),axis = 1).tolist()
    #        compute DF_pair
            dfpairList = []
            for i in range( len ( self.candidate_index ) ):
                dfpair = heapq.nlargest( 2, self.prob[ i, : ])
                a = max( dfpair[ 0 ], dfpair[ 1 ] )
                b = min( dfpair[ 0 ], dfpair[ 1 ] )
                dfpair = a-b   
                dfpairList.append((self.lam-self.al_count*self.lam_ur)*dfpair )
    #        compute DF_all
            dfallList = []
            for i in range( len ( self.candidate_index ) ):
                #dfall = sp.stats.dfall( self.prob[ i, : ].T )
                # 原代码: dfall = sp.stats.dfall( self.prob[ i, : ].T )
                dfall = sp.stats.entropy( self.prob[ i, : ].T )   
                if self.al_count>0.1*self.M:
                    dfallList.append(dfall )
                else:
                    dfallList.append(0)
                     
            mcList = list(map(operator.sub, dfpairList, dfallList))
            dfList = []
            for i in range(len(mcList)):
                if self.al_count>0.1*self.M:
                    dfGam = (self.gam-self.al_count*self.gam_ur)*resultList[i]
                    if dfGam>0:                    
                        dfList.append(dfGam)
                    else:
                        dfList.append(0)
                else:
                    dfList.append((self.al_count)*self.gam_ur*resultList[i])
        
            resultList = list(map(operator.add, dfList, mcList))
            targetIndexList = list(map(resultList.index, \
                                       heapq.nsmallest(n_batch, resultList ) ))
        else:
        # F_rank sampling
            resultList = []
            for i in range(len(Xgrid)):
                dis = dissonance(dec_Train[i,:])
                resultList.append(dis)
            targetIndexList = list(map(resultList.index,heapq.nlargest(n_batch, resultList ) ))
        delList = []
        
        for targetIndex in targetIndexList:            
            delList.append(self.candidate_index[ targetIndex ])
            self.train_index.append( self.candidate_index[ targetIndex ] )
        self.candidate_index = [x for x in self.candidate_index \
                                if x not in delList]
#        retrain model
        self.clf.fit( self.data[ self.train_index, : ] , \
                     self.Y[ self.train_index])
#        update prob
        self.prob = self.clf.predict_proba( \
                            self.data[ self.candidate_index , : ])
        return 0
    
    def evaluate(self):
#        compute current test accuracy
        self.currentPerformance = \
            self.clf.score( self.data[ self.test_index, : ] , \
                           self.Y[ self.test_index ] );
        return  self.currentPerformance