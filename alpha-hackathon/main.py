from config import config
from utils import DataWork
import models
import time

dw = DataWork()

def main():
    start = time.monotonic()
    train = config.panel.train
    submit = config.panel.train
    data_train, data_test = dw()
    df_val = dw.df_val
    model = getattr(models, config.panel.model)()
    if train:
        ndcg, preds = model.run(
            data = data_train,
            data_test = data_test,
            df_val = df_val
        )
        print(f'NDCG@5: {ndcg}')
    else:
        preds = model.predict(data = data_test)
    if submit:
        dw.submit(name = config.panel.model, preds_score = preds)
    end = time.monotonic()
    print(f'Script completed by {end-start} second')

if __name__ == '__main__':
    main()