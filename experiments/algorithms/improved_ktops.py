from sklearn.cluster import KMeans

class sample(object):
    def __init__(self, clf, data, Y, train_index, candidate_index, test_index, M,
                 gam, gam_ur, lam, lam_ur, gam_clf=1, use_sv_dedup=True, dedup_clusters=10):
        # ... 初始化（添加去重参数）...
        self.use_sv_dedup = use_sv_dedup
        self.dedup_clusters = dedup_clusters
    
    def dtrustSample(self, n_batch=1):
        self.al_count += 1
        # ... 计算dec_Train等...
        
        if dualAve_all < 1.5*self.dualAve_all_0:
            # 线性衰减（原始）
            lam_weight = max(0.0, self.lam - self.al_count*self.lam_ur)
            gam_weight = max(0.0, self.gam - self.al_count*self.gam_ur)
            
            resultList = np.amax(dec_Train/np.mean(sp_nList), axis=1)
            
            final_scores = []
            for i in range(len(self.candidate_index)):
                probs = self.prob[i,:]
                dfpair = heapq.nlargest(2, probs)
                margin = (dfpair[0] - dfpair[1]) if len(dfpair)>1 else 0
                ent = sp.stats.entropy(probs)
                score = (gam_weight * resultList[i]) + (lam_weight * margin) - ent
                final_scores.append(score)
            
            # Top-K去重（核心改进）
            if self.use_sv_dedup and len(final_scores) > 0:
                try:
                    n_cand = len(final_scores)
                    n_clusters = min(self.dedup_clusters, n_cand)
                    if n_clusters > 0:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dec_Train)
                        labels = kmeans.labels_
                        cluster_reps = []
                        for cl in range(n_clusters):
                            members = [idx for idx, lab in enumerate(labels) if lab == cl]
                            if members:
                                best_member = min(members, key=lambda idx: final_scores[idx])
                                cluster_reps.append(best_member)
                        
                        cluster_reps = list(set(cluster_reps))
                        targetIndexList = sorted(cluster_reps, key=lambda idx: final_scores[idx])[:n_batch]
                        if len(targetIndexList) < n_batch:
                            chosen = set(targetIndexList)
                            remaining = [i for i in range(len(final_scores)) if i not in chosen]
                            more = sorted(remaining, key=lambda idx: final_scores[idx])[:(n_batch-len(targetIndexList))]
                            targetIndexList.extend(more)
                    else:
                        targetIndexList = list(np.argsort(final_scores)[:n_batch])
                except:
                    targetIndexList = list(np.argsort(final_scores)[:n_batch])
            else:
                targetIndexList = list(np.argsort(final_scores)[:n_batch])
        else:
            # Ranking阶段不变
            pass