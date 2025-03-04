from preprocess.date_converter import DateConverter
from sklearn.pipeline import Pipeline
from preprocess.nan_fixer import  CustomImputer
import pandas as pd
from preprocess.encoding import FrecuencyEncoding
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from preprocess.scaler import CustomScaler
import numpy as np




def basic_preprocess(df, target : str):

    important_features = ['amt', 'category', 'merchant', 'trans_date_trans_time', 'unix_time', 'dob', 'street', 'merch_lat', 'merch_long', 'city', 'merch_zipcode', 'city_pop', 'job', 'last', 'first', 'cc_num', 'long', 'zip', "is_fraud"]

    df = df.drop(["trans_num", "Unnamed: 0"], axis=1)
    df = df[important_features]

    df_train, unseen_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df[target])
    df_val, df_test = train_test_split(unseen_df, test_size=0.5, shuffle=True, random_state=42, stratify=unseen_df[target])

    pipeline = Pipeline([
        ("imputer",         CustomImputer(strategy="mean", attributes=["merch_zipcode"])),
        ("date_converter",  DateConverter(["trans_date_trans_time", "dob"])),
        ("encoding",        FrecuencyEncoding()),
        ("scaler",          CustomScaler(df.drop(target, axis=1).columns.tolist())),
        ("pca",             PCA(n_components=0.999)),
    ], verbose=True)
    

    X_train = pd.DataFrame(pipeline.fit_transform(df_train.drop(target, axis=1), df_train[target]),   index=df_train.index)
    X_test  = pd.DataFrame(pipeline.transform(df_test.drop(target, axis=1)),    index=df_test.index)
    X_val   = pd.DataFrame(pipeline.transform(df_val.drop(target, axis=1)),     index=df_val.index)


    df_train    = pd.concat([X_train, df_train[target]], axis=1)
    df_test     = pd.concat([X_test, df_test[target]], axis=1)
    df_val      = pd.concat([X_val, df_val[target]], axis=1)

    
    return [df_train, df_test, df_val]
