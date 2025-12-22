# 实验配置
EXPERIMENT_CONFIG = {
    'datasets': ['yeast', 'dermatology', 'vehicle'],
    'algorithms': {
        'baseline': {
            'name': 'Baseline (原始)',
            'params': {
                'gam': 1.0, 'gam_ur': 0.005, 
                'lam': 1.0, 'lam_ur': 0.01,
                'use_sv_dedup': False
            }
        },
        'improved_init': {
            'name': '仅改进初始化',
            'params': {
                'gam': 1.0, 'gam_ur': 0.005,
                'lam': 1.0, 'lam_ur': 0.01,
                'use_sv_dedup': False
            }
        },
        'improved_decay': {
            'name': '仅改进衰减',
            'params': {
                'gam': 1.0, 'gam_ur': 0.02,  # 注意：指数衰减用不同参数
                'lam': 1.0, 'lam_ur': 0.02,
                'use_sv_dedup': False
            }
        },
        'improved_diversity': {
            'name': '仅改进去重',
            'params': {
                'gam': 1.0, 'gam_ur': 0.005,
                'lam': 1.0, 'lam_ur': 0.01,
                'use_sv_dedup': True,
                'dedup_clusters': 20
            }
        },
        'improved_full': {
            'name': '完整改进',
            'params': {
                'gam': 1.0, 'gam_ur': 0.02,
                'lam': 1.0, 'lam_ur': 0.02,
                'use_sv_dedup': True,
                'dedup_clusters': 20
            }
        }
    },
    'experiment': {
        'n_initial': 10,      # 初始样本数
        'n_queries': 150,     # 查询次数
        'n_repeats': 5,       # 重复次数（统计显著性）
        'test_size': 0.3      # 测试集比例
    }
}