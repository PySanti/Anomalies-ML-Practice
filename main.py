import pandas as pd
from sklearn.metrics import f1_score
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV




TARGET = "is_fraud"
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_val = pd.read_csv("./data/val.csv")


param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_samples': [0.3, 0.45, 0.5, 0.65],
    'contamination': [0.01, 0.05, 0.1, "auto"],
    'max_features': [0.5, 0.75, 1.0],
    'bootstrap': [True, False],
    'random_state': [42]
}

def custom_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    return f1_score(y, y_pred, pos_label=1)

grid_search = RandomizedSearchCV(
    estimator=IsolationForest(),
    param_distributions=param_grid,
    scoring=custom_scorer,
    n_iter = 50,
    cv=3,
    n_jobs=4,
    verbose=10
)

grid_search.fit(df_train.drop(TARGET, axis=1), df_train[TARGET])
joblib.dump(grid_search.best_estimator_, "if.joblib")

print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor precision:", grid_search.best_score_)
