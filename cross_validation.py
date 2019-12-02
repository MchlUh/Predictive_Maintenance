import random


class cross_val:
    def __init__(self, model, df, folds):
        self.model = model
        self.df = df
        self.folds = folds
        self.cv_results = None

    def split_dataset(self):
        engine_ids = list(set(self.df.index))
        random.shuffle(engine_ids)
        fold_length = len(engine_ids)//self.folds
        engine_folds = []
        for i in range(self.folds - 1):
            engine_folds.append(engine_ids[i*fold_length : (i+1)*fold_length])
        engine_folds.append(engine_ids[(self.folds - 2) * fold_length:])

    def cross_validate(self):

        return 1

k = cross_val('model', 'df', 5)