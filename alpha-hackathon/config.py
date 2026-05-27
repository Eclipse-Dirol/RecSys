from omegaconf import OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_config = {
    'panel':{
        'model': ['LGBM'], # only LGBM and XGB
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
        'folds': 5,
    },
    'get_model': {
        "LGBM": {
            "module": "models.boosting",
            "class": "LGBM",
        },
        "XGB": {
            'module':  "models.boosting",
            "class": "XGB",
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
        'xgb': {
            'objective': "rank:ndcg",
            'eval_metric': "ndcg@5",
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'tree_method': "hist",
            'random_state': 42,
            'n_jobs': -1,
            'enable_categorical': True
        },
    }
}

config = OmegaConf.create(data_config)