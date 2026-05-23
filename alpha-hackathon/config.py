from omegaconf import OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_config = {
    'panel':{
        'model': 'LGBM', # only LGBM and two-tower
        'submit': True,
        'train': True,
        'save_model': True
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
        'drop_feat': ['app_id', 'request_received', 'date_part', 'request_id', 'offer_id'],
        'test_size': 0.2,
        'random_state': 42,
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
    }
}

config = OmegaConf.create(data_config)