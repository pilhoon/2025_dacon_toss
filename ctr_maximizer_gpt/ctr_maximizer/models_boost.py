
from typing import Dict, Any, Optional
import numpy as np

def train_xgboost(X_train, y_train, X_valid, y_valid, sample_weight=None, params: Optional[Dict[str, Any]] = None):
    import xgboost as xgb
    params = params or {}
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    booster_params = {
        'max_depth': 8,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss','aucpr'],
        'tree_method': 'hist',
    }
    booster_params.update(params)
    evallist = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(booster_params, dtrain, num_boost_round=2000, evals=evallist,
                    early_stopping_rounds=100, verbose_eval=100)
    def predict(X):
        return bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration+1))
    return bst, predict

def _auto_cat_features(X_train, max_card: int = 1000, sample_n: int = 100000, random_state: int = 42):
    import pandas as pd
    n = len(X_train)
    samp = X_train.sample(n=min(n, sample_n), random_state=random_state) if n > 0 else X_train
    cat_idx = []
    for i, c in enumerate(X_train.columns):
        s = samp[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            cat_idx.append(i)
        elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= max_card:
            cat_idx.append(i)
    return cat_idx

def train_catboost(X_train, y_train, X_valid, y_valid, sample_weight=None, params: Optional[Dict[str, Any]] = None):
    import pandas as pd
    from catboost import CatBoostClassifier, Pool
    params = params or {}
    defaults = dict(
        depth=8,
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        loss_function='Logloss',
        eval_metric='PRAUC',
        iterations=20000,
        random_seed=42,
        od_type='Iter',
        od_wait=200,
        verbose=100
    )
    defaults.update(params or {})
    cat_idx = defaults.pop('cat_features', None)
    if cat_idx is None:
        cat_idx = _auto_cat_features(X_train)
    train_pool = Pool(X_train, y_train, weight=sample_weight, cat_features=cat_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx)
    model = CatBoostClassifier(**defaults)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=100)
    def predict(X):
        return model.predict_proba(X)[:,1]
    return model, predict
