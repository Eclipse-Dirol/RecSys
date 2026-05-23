import pandas as pd
import numpy as np
from config import config
import math
import torch
from sklearn.model_selection import train_test_split

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
        pass

    def __call__(self):
        return self.run()

    def __getattribute__(self, name):
        if name == 'df_val':
            return super().__getattribute__('df_val')
        return super().__getattribute__(name)

    def load_data(
        self,
        path: str = None,
        extension: str = None
    ):
        if extension == 'pq':
            return pd.read_parquet(path)
        if extension == 'csv':
            return pd.read_csv(path)

    def prep(
        self,
        data_ed: pd.DataFrame = None,
        data_feat: pd.DataFrame = None,
        submit: bool = False,
        train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray] | pd.DataFrame:
        drop_feat = config.args.drop_feat
        target = config.args.target
        data_ed_temp = data_ed.copy()
        data_feat_temp = data_feat.copy()

        data_ed_temp.ncl = pd.to_numeric(data_ed_temp.ncl)
        data_ed_temp.eva = pd.to_numeric(data_ed_temp.eva)
        data_ed_temp.eva_perc = pd.to_numeric(data_ed_temp.eva_perc)
        data_ed_temp.rate = pd.to_numeric(data_ed_temp.rate)
        
        all_data= pd.merge(data_ed_temp, data_feat_temp, on=['app_id', 'date_part'])
        cat_cols = all_data.select_dtypes(include=['object'])
        for col in cat_cols:
            all_data[col] = all_data[col].astype('category')

        if submit is True and train is not True:
            self.request_id = all_data.request_id
            self.variant_no = all_data.variant_no
            return all_data.drop(columns=drop_feat)

        tr_group_id, val_group_id = train_test_split(all_data.request_id.unique(),
                                                    test_size=config.args.test_size,
                                                    random_state=config.args.random_state)

        df_train = all_data[all_data.request_id.isin(tr_group_id)].sort_values('request_id').copy()
        self.df_val = all_data[all_data.request_id.isin(val_group_id)].sort_values('request_id').copy()

        tr_group = df_train.groupby("request_id", sort=False).size().to_numpy()
        val_group = self.df_val.groupby("request_id", sort=False).size().to_numpy()

        X_tr, X_val = df_train.drop(columns=drop_feat+[target]).reset_index(drop=True), self.df_val.drop(columns=drop_feat+[target]).reset_index(drop=True)
        y_tr, y_val = df_train[target], self.df_val[target]

        return (X_tr, y_tr, X_val, y_val, tr_group, val_group)

    def run(self) -> tuple[tuple, pd.DataFrame | None]:
        submit = config.panel.submit
        train = config.panel.train
        data_feat = self.load_data(
            path = config.path.features[0],
            extension= config.path.features[1]
        )
        data_train = self.load_data(
            path = config.path.train[0],
            extension= config.path.train[1]
        )
        data_train = self.prep(
            data_ed = data_train,
            data_feat = data_feat,
            submit = submit,
            train=train
        )
        if submit:
            data_test = self.load_data(
                path = config.path.test[0],
                extension= config.path.test[1]
            )
            data_test = self.prep(
                data_ed = data_test,
                data_feat = data_feat,
                submit = submit,
                train=False
            )
            return (data_train, data_test)
        return (data_train, None)

    def submit(
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