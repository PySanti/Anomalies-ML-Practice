import pandas as pd
from utils.basic_preprocess import basic_preprocess


TARGET = "is_fraud"
[df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), target=TARGET)

print(df_train[TARGET].value_counts())
print(df_test[TARGET].value_counts())
print(df_val[TARGET].value_counts())
