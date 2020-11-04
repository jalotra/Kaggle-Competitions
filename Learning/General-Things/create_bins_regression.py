''' To make starified KFolds in a Regression Problem is the output target values are not 
distributed over {R}''' 

import numpy as np
import pandas as pd
from pprint import pprint

from sklearn import model_selection, datasets

# data is a pandas DataFrame
def create_folds(data):
    data["kfold"] = -1
    data = data.sample(frac = 1).reset_index(drop = True)
    assert len(data) != 0
    num_bins = np.floor(1 + np.log2(len(data)))

    data.loc[:, "bins"] = pd.cut(
        data["target"], bins = num_bins, labels = False
    )

    kf = model_selection.StratifiedKFold(n_splits = 5)

    for fold, (train, validate) in enumerate(kf.split(X = data, y = data.bins.values)):
        data.loc[validate, "kfold"] = fold

    # Drop the bins columns
    data = data.drop("bins", axis = 1)

    return data

if __name__ == "__main__":

    X, y = datasets.make_regression(
        n_samples = 10000, n_features = 50, n_targets = 1
    )


    # Create a Pandas DataFrame

    df = pd.DataFrame(
        X, columns = [f"shivam_{number}" for number in range(X.shape[1])]
    )
    df.loc[:, "target"] = y

    df = create_folds(df)
    
    pprint(df.head())













