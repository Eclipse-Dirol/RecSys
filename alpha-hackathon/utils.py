import pandas as pd
import numpy as np
from config import config
import math
from catboost import Pool
from sklearn.model_selection import train_test_split
from importlib import import_module

class NDCG():
    def __init__(self):
        self.k = config.args.k

    def ndcg_at_k(self, rel):
        """DCG@k для бинарной релевантности (0/1)"""
        top_k = rel[:self.k]
        dcg = sum(1.0 / math.log2(i+2) for i, g in enumerate(top_k) if g == 1)

        ideal_rel = sorted(rel, reverse=True)[:self.k]
        idcg = sum(1.0 / math.log2(i+2) for i, g in enumerate(ideal_rel) if g == 1)

        return dcg / idcg if idcg > 0 else np.nan

    def mean_ndcg_at_5(self, df, request_col='request_id', score_col='score', target_col='is_deal'):
        """
        Вычисляет средний NDCG@5 по всем request_id в датафрейме.

        Параметры
        ----------
        df : pd.DataFrame
        Колонки: request_id, deal, score, target (0/1).
        Для каждого request_id может быть от 1 до 50 строк (предложений),
        и ровно одна строка с target=1 (покупка).
        request_col : str
        score_col : str
        target_col : str

        Возврат
        -------
        float : средний NDCG@5
        """
        # Сортируем внутри каждого request_id по убыванию скоров модели
        df_sorted = df.sort_values([request_col, score_col], ascending=[True, False])

        # Группируем и считаем NDCG@5
        ndcg_per_request = df_sorted.groupby(request_col)[target_col].apply(
        lambda x: self.ndcg_at_k(x.tolist())
        )
        return np.nanmean(ndcg_per_request)

class DataWork():
    def __init__(self):
        self.submit = config.panel.submit
        self.train = config.panel.train
        self.drop_feat = config.args.drop_feat
        self.target = config.args.target

    def __call__(self, name):
        return self.run(name)

    def read_data(
        self,
        path: str = None,
        extension: str = None
    ):
        if extension == 'pq':
            return pd.read_parquet(path)
        if extension == 'csv':
            return pd.read_csv(path)

    def load_data(self):
        data = {}
        if config.panel.train:
            train_data = self.read_data(
                path = config.path.train[0],
                extension = config.path.train[1],
            )
            data['train'] = train_data
        if config.panel.submit:
            test_data = self.read_data(
                path = config.path.test[0],
                extension = config.path.test[1],
            )
            data['test'] = test_data
        feat_data = self.read_data(
            path = config.path.features[0],
            extension = config.path.features[1],
        )
        data['feat'] = feat_data
        return data

    def universal_prep(
        self,
        data_ed: pd.DataFrame = None,
        data_feat: pd.DataFrame = None,
    ):
        data_ed_temp = data_ed.copy()
        data_feat_temp = data_feat.copy()

        data_ed_temp.ncl = pd.to_numeric(data_ed_temp.ncl)
        data_ed_temp.eva = pd.to_numeric(data_ed_temp.eva)
        data_ed_temp.eva_perc = pd.to_numeric(data_ed_temp.eva_perc)
        data_ed_temp.rate = pd.to_numeric(data_ed_temp.rate)
        
        all_data= pd.merge(data_ed_temp, data_feat_temp, on=['app_id', 'date_part'])

        cat_cols = all_data.select_dtypes(include=['object', str]).columns.tolist()
        for col in cat_cols:
            all_data[col] = all_data[col].astype('category')

        all_data['date_part'] = pd.to_datetime(all_data['date_part'])
        all_data["day"] = all_data["date_part"].dt.day
        all_data["weekday"] = all_data["date_part"].dt.weekday
        all_data["month"] = all_data["date_part"].dt.month
        all_data["week"] = all_data["date_part"].dt.isocalendar().week.astype(int)
        all_data["is_month_start"] = all_data["date_part"].dt.is_month_start.astype(int)
        all_data["is_month_end"] = all_data["date_part"].dt.is_month_end.astype(int)

        return all_data

    def prep_for_model(
        self,
        data: dict = None,
        name: str = None,
    ) -> tuple:
        tr_group_id, val_group_id = train_test_split(data.request_id.unique(),
                                                    test_size=config.args.test_size,
                                                    random_state=config.args.random_state)
        
        df_train = data[data.request_id.isin(tr_group_id)].sort_values('request_id').copy()
        df_val = data[data.request_id.isin(val_group_id)].sort_values('request_id').copy()

        X_tr = df_train.drop(columns=self.drop_feat + [self.target]).reset_index(drop=True)
        X_val = df_val.drop(columns=self.drop_feat + [self.target]).reset_index(drop=True)

        y_tr = df_train[self.target].reset_index(drop=True)
        y_val = df_val[self.target].reset_index(drop=True)

        tr_group_id = df_train['request_id'].reset_index(drop=True)
        val_group_id = df_val['request_id'].reset_index(drop=True)

        match name:
            case 'LGBM':
                tr_group = df_train.groupby('request_id', sort=False).size().to_numpy()
                val_group = df_val.groupby('request_id', sort=False).size().to_numpy()

                return (X_tr, y_tr, X_val, y_val, tr_group, val_group, df_val)

            case 'NN':
                pass

    def run(self, name_model: str = None) -> tuple:
        data_ed = self.load_data()
        all_data = {}
        for name in data_ed.keys():
            if name == 'feat':
                continue
            data = data_ed[name]
            all_data[name] = self.universal_prep(data, data_ed['feat'])
        self.request_id = all_data['test']['request_id']
        self.variant_no = all_data['test']['variant_no']
        all_data['test'] = all_data['test'].drop(columns=self.drop_feat).reset_index(drop=True)
        if self.train:
            data = self.prep_for_model(
                data = all_data['train'],
                name = name_model
            )
            return (data, all_data['test'])
        return (all_data['test'])

    def save_preds(
            self, 
            name: str = None,
            preds_score: np.ndarray = None,
        ):
        submit = pd.DataFrame({
            'request_id': self.request_id,
            'variant_no': self.variant_no,
            'score': preds_score,
        })
        submit.to_csv(f'{config.path.submit}/{name}.csv', index=False, sep=';')

class ModelFactory():
    @staticmethod
    def get_model(name: str):
        if name not in config.get_model:
            raise ValueError(f"Unknown model name: {name}")

        model_info = config.get_model[name]

        module_path = model_info["module"]
        class_name = model_info["class"]

        module = import_module(module_path)
        model_cls = getattr(module, class_name)

        return model_cls()