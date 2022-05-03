"""
Taken from https://github.com/optuna/optuna-examples

"""

from random import seed
import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from dask_ml.model_selection import train_test_split
from dask.distributed import Client, LocalCluster
from optuna.samplers import RandomSampler
import xgboost as xgb
from optuna.pruners import MedianPruner
import dask.array as da
import joblib
import time

SEED = 108
# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial, client):

    # Load our dataset
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train, y_train = da.from_array(X_train, chunks=len(X_train) // 5), da.from_array(y_train, chunks=len(y_train) // 5)
    X_test, y_test = da.from_array(X_test, chunks=X_train.chunksize), da.from_array(y_test, chunks=y_train.chunksize)
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    dvalid = xgb.dask.DaskDMatrix(client, X_test, y_test)
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        # Suggest a value for the categorical parameter.https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        # Analogous to learning rate in GBM
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # amma specifies the minimum loss reduction required to make a split.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    # bst = xgb.dask.train(client, param, dtrain, evals=[(dvalid, "validation")], \
    #      num_boost_round=trial.suggest_int("num_boosting_rounds", 1, 100))
    bst = xgb.dask.train(client, param, dtrain, evals=[(dvalid, "validation")], \
         num_boost_round=trial.suggest_int("num_boosting_rounds", 1, 100), callbacks=[pruning_callback])

    preds = xgb.dask.predict(client, bst, dvalid).persist()
    # preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_test, pred_labels)
    return accuracy

def execute_optimization(study_name, client, trials,
                         params=dict(), direction='maximize'):
    
    ## We use pruner to skip trials that are NOT fruitful
    pruner = MedianPruner(n_warmup_steps=5)
    
    study = optuna.create_study(direction=direction,
                         study_name=study_name,
                         storage='sqlite:///optuna.db',
                         load_if_exists=True,
                         pruner=pruner)
    with joblib.parallel_backend("dask"):
        study.optimize(lambda trial: objective(trial, client), \
            n_trials=trials, n_jobs=-1)
        
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    
    return study.best_params
if __name__ == "__main__":

    clusters = LocalCluster()
    client = Client(clusters)
    print(client)
    time.sleep(10)
    start_time = time.time()
    study = optuna.create_study(sampler=RandomSampler(seed=102),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(lambda trial: objective(trial, client), n_trials=100)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("--- %s seconds ---" % (time.time() - start_time))
    # execute_optimization('xgboost', client, 100, direction='maximize')


