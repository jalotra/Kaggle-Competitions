import numpy as np
import pandas as pd
import config

from pprint import pprint
from sklearn import model_selection

def create_folds(df):
    df["kfold"] = -1
    df = df.sample(frac = 1).reset_index(drop = True)
    assert len(df) != 0

    kf = model_selection.StratifiedKFold(n_splits = 5)

    for fold, (train, validate) in enumerate(kf.split(X = df, y = df.Activity.values)):
        df.loc[validate, "kfold"] = fold

    return df


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_CSV)
    df = create_folds(df)
    df.to_csv(config.SAVE_STRATIFIED)


