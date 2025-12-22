class sample(object):
    # ... 与baseline相同的初始化 ...
    
    def dtrustSample(self, n_batch=1):
        self.al_count += 1
        # ... 计算dec_Train等（与baseline相同）...
        
        if dualAve_all < 1.5*self.dualAve_all_0:
            # 指数衰减（核心改进）
            decay_factor = np.exp(-0.02 * self.al_count)  # 指数衰减
            current_lam = self.lam * decay_factor
            current_gam = self.gam * decay_factor
            
            resultList = np.amax(dec_Train/np.mean(sp_nList), axis=1)
            
            final_scores = []
            for i in range(len(self.candidate_index)):
                probs = self.prob[i,:]
                dfpair = heapq.nlargest(2, probs)
                margin = (dfpair[0] - dfpair[1]) if len(dfpair)>1 else 0
                ent = sp.stats.entropy(probs)
                
                # 使用指数衰减的权重
                score = (current_gam * resultList[i]) + (current_lam * margin) - ent
                final_scores.append(score)
            
            # 不使用去重
            targetIndexList = list(np.argsort(final_scores)[:n_batch])
        else:
            # Ranking阶段保持不变
            pass