import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500,class_sep=0.7)
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.33, random_state=42)

data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
params = {'objective':'binary:logistic','eval_metric':'logloss',
          'eta':0.01,
          'subsample':0.1,
          'num_boost_rounds':100}
xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5, metrics = 'logloss',seed=42)
print(xgb_cv)
