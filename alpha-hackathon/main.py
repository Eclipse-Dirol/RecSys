from config import config
from utils import DataWork, ModelFactory
import time

dw = DataWork()

def main():
    start = time.monotonic()
    train = config.panel.train
    submit = config.panel.train
    score = []
    for model_name in config.panel.model:
        data_all = dw(model_name)
        ranker_model = ModelFactory.get_model(model_name)
        if train:
            result, preds = ranker_model.run(
                data = data_all[0],
                data_test = data_all[1]
            )
            if config.panel.show_log:
                print('log:')
                print(f'model fit_time: {result["fit_time"]}')
                print(f'model eval_time: {result["eval_time"]}')
            print(f'----------NDCG@5 by {model_name}: {result["ndcg"]}')
            score.append(preds)
        else:
            preds = model_name.predict(data = data_all[1])
        if submit and len(config.panel.model) < 2:
            dw.save_preds(name = model_name, preds_score = preds)
        end = time.monotonic()
        if config.panel.show_log:
            print(f'Script completed by {end-start} second')
    if len(config.panel.model) > 1:
        final_score = 0
        for i in range(len(config.panel.model)):
            final_score += score[i] * config.panel[f'coef_{i+1}']
        if submit:
            dw.save_preds(name = 'ensemble', preds_score = final_score)

if __name__ == '__main__':
    main()