from config import config
import pandas as pd
import numpy as np
from lightgbm import LGBMRanker, early_stopping
from utils import NDCG, DataWork
import joblib
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

metrics = NDCG()

class LGBM():
    def __init__(self):
        self.train = config.panel.train
        self.submit = config.panel.submit
        param = dict(config.param.lgbm)
        self.model = LGBMRanker(**param, ndcg_eval_at = [5])
        self.save = config.panel.save_model

    def model_training(
        self,
        X_tr: pd.DataFrame = None,
        y_tr: pd.DataFrame = None,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None,
        tr_group: np.ndarray = None,
        val_group: np.ndarray = None,
        df_val: pd.DataFrame = None,
    ):
        start_fit = time.monotonic()
        self.model.fit(
            X_tr, y_tr,
            group=tr_group,
            eval_set=[(X_val, y_val)],
            eval_group=[val_group],
            callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
        )
        end_fit = time.monotonic()
        start_ndcg = time.monotonic()
        pred = self.model.predict(X_val)
        df_val_eval = df_val.copy()
        df_val_eval["score"] = pred
        ndcg_5 = metrics.mean_ndcg_at_5(
            df_val_eval,
            request_col="request_id",
            score_col="score",
            target_col="is_deal"
        )
        if self.save:
            self.save_model()
        end_ndcg = time.monotonic()
        print(f'model fited by {end_fit-start_fit} second')
        print(f'preds, save and calculate metrics completed by {end_ndcg-start_ndcg} second')
        return ndcg_5

    def predict(
        self,
        data: pd.DataFrame = None
    ):
        model = self.load_model()
        return model.predict(data)

    def save_model(self):
        joblib.dump(self.model, f'{config.path.save}/lgbm-ranker.pkl')
        return self

    def load_model(self) -> any:
        return joblib.load(f'{config.path.load}/lgbm-ranker.pkl')

    def run(
        self,
        data: tuple = None,
        data_test: pd.DataFrame | None = None,
        df_val: pd.DataFrame | None = None,
    ) -> tuple:
        ndcg = self.model_training(
            X_tr=data[0],
            y_tr=data[1],
            X_val=data[2],
            y_val=data[3],
            tr_group=data[4],
            val_group=data[5],
            df_val=df_val
        )
        if self.submit:
            preds = self.predict(data_test)
        return (ndcg, preds)