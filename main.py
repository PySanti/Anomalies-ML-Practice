import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import joblib
from utils.gmm_utils import custom_scorer
from utils.gmm_utils import generate_filename

from utils.basic_preprocess import basic_preprocess


TARGET = "is_fraud"


param_grid = {  
    'n_components': [2, 3, 4, 5, 6, 7, 8],
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'max_iter': [100, 200, 300],
    'tol': [1e-3, 1e-4, 1e-5]
}  


for scaler, pca in [(True, True), (True, False), (False, False), (False, True)]:
    filename = generate_filename(scaler, pca)
    print(f"Generando {filename}")
    [df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/original_data.csv"), TARGET, scaler=scaler, pca=pca)
    grid_search = GridSearchCV(GaussianMixture(), param_grid, cv=4, n_jobs=6, verbose=10, scoring=custom_scorer(df_val, TARGET))
    grid_search.fit(df_train.query(f"{TARGET} == 0").drop(TARGET, axis=1))
    print(f"Mejor precision {grid_search.best_score_}")
    print(f"Mejores hiperparametros {grid_search.best_params_}")
    joblib.dump(grid_search.best_estimator_, filename)

