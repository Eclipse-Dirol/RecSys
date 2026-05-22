from omegaconf import OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    'path': {
        'train': [f'{BASE_DIR}/data/train_dataset_small.pq', 'pq'],
        'test': [f'{BASE_DIR}/data/test_dataset_small.pq', 'pq'],
        'feateres': [f'{BASE_DIR}/data/feateres_small.pq', 'pq'],
        'discription': [f'{BASE_DIR}/data/feature_description.csv', 'csv'],
    },
}

config = OmegaConf.create(config)