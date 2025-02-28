import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from time import time
import numpy as np
import joblib




TARGET = "is_fraud"
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_val = pd.read_csv("./data/val.csv")



def custom_scorer(estimator, X):
    log_probs_val = estimator.score_samples(df_val.drop(TARGET, axis=1))
    best_score = 0
    for a in np.linspace(log_probs_val.min(), log_probs_val.max(), 100):
        y_pred = (log_probs_val < a).astype(int)
        f1 = f1_score(df_val[TARGET], y_pred, pos_label=1)
        if f1 > best_score:
            best_score = f1
    return best_score

param_grid = {  
    'n_components': [2],  # Número de componentes  
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Tipos de covarianza  
    'max_iter': [100, 200, 300],  # Número máximo de iteraciones  
    'tol': [1e-3, 1e-4, 1e-6]  # Tolerancia  
}

grid_search = GridSearchCV(GaussianMixture(), param_grid, cv=3, n_jobs=4, verbose=10, scoring=custom_scorer)
grid_search.fit(df_train.drop(TARGET, axis=1))
print(grid_search.best_score_)

joblib.dump(grid_search.best_estimator_, "gmm.joblib")
