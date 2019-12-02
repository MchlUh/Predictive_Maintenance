import random


class CrossVal:
    def __init__(self, model, df, n_folds):
        self.model = model
        self.df = df
        self.n_folds = n_folds
        self.cv_results = None
        self.folds = None

    def split_dataset(self):
        engine_ids = list(set(self.df.index))
        random.shuffle(engine_ids)
        fold_length = len(engine_ids)//self.n_folds
        engine_folds = []
        for i in range(self.n_folds - 1):
            engine_folds.append(engine_ids[i*fold_length : (i+1)*fold_length])
        engine_folds.append(engine_ids[(self.n_folds - 2) * fold_length:])
        self.folds = engine_folds

    def cross_validate(self):
        self.split_dataset()

        return 1


k = CrossVal('model', 'df', 5)
