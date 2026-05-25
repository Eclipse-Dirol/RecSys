from omegaconf import OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_config = {
    'panel':{
        'model': ['LGBM'], # only LGBM and Catboost
        'submit': True,
        'train': True,
        'save_model': True,
        'show_log': True,
        'coef_1': 0.5, # first model have this coef
        'coef_2': 0.5 # second model have this coef, third model have coef = 1 - coef_1 - coef_2
    },
    'path': {
        'train': [f'{BASE_DIR}/data/train_dataset_small.pq', 'pq'],
        'test': [f'{BASE_DIR}/data/test_dataset_small.pq', 'pq'],
        'features': [f'{BASE_DIR}/data/features_small.pq', 'pq'],
        'discription': [f'{BASE_DIR}/data/feature_description.csv', 'csv'],
        'save': f'{BASE_DIR}/data/models',
        'load': f'{BASE_DIR}/data/models',
        'submit': f'{BASE_DIR}/data/submit'
    },
    'args': {
        'k': 5,
        'group_id': 'request_id',
        'target': 'is_deal',
        'drop_feat': ['app_id', 'request_received', 'request_id', 'offer_id', 'date_part'],
        'test_size': 0.2,
        'random_state': 42,
    },
    'get_model': {
        "LGBM": {
            "module": "models.boosting",
            "class": "LGBM",
        },
        "Catboost": {
            "module": "models.boosting",
            "class": "CatBoost",
        },
        "NN": {
            "module": "models.nn",
            "class": "NN",
        },
    },
    'param': {
        'lgbm': {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'n_estimators': 1000,
            'subsample': 0.8,
            'bagging_freq': 1,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs':-1,
            'verbose': -1
        },
        'catboost': {
            'loss_function': 'YetiRank',
            'eval_metric': 'NDCG:top=5',
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 10,
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'verbose': 0,
            'allow_writing_files': False
        },
    }
}

config = OmegaConf.create(data_config)