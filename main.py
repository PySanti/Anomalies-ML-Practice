import pandas as pd
from utils.basic_preprocess import basic_preprocess

[df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), target="is_fraud")

