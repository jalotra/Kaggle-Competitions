import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import config
from pprint import pprint
import argparse

def run(fold):

    df = pd.read_csv(config.SAVE_STRATIFIED)

    # Load the test and train dataframes
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    #Convert the dataframe to numpy array
    # Train arrays
    # Validation arrays
    x_train = df_train.drop("Activity", axis = 1).values
    y_train = df_train.Activity.values

    x_valid = df_valid.drop("Activity", axis = 1).values
    y_valid = df_valid.Activity.values

    predictions = model(x_train, y_train, x_valid)
    loss = metrics.log_loss(predictions, y_valid)

    print(f"The loss on {fold} fold is : ", loss)


def model(x_train, y_train, x_valid):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    predictions = clf.predict_proba(x_valid)
    print(clf.get_depth())   

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )

    args = parser.parse_args()
    run(fold = args.fold)
