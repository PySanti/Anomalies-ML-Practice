import pandas as pd
from sklearn.metrics import f1_score
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from utils.basic_preprocess import basic_preprocess




TARGET = "is_fraud"
[df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/original_data.csv"), TARGET)

df_train.to_csv("./pca_data/train_set.csv")
df_test.to_csv("./pca_data/test_set.csv")
df_val.to_csv("./pca_data/val_set.csv")
