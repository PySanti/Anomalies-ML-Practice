# Anomalies

El objetivo de este proyecto sera poner en practica nuestros conocimientos recientes de aprendizaje no supervisado y semi-supervisado tomando un dataset desequilibrado (producto del estudio de anomalias), y revisar cual de los siguientes algoritmos genera mejores resultados:

* K-MEANS
* DB-SCAN
* Isolation Forest
* GMM

El [dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) que trataremos contiene informacion acerca de transacciones bancarias, de las cuales, el 0,57% son fraude. El objetivo sera tratar de identificar relaciones entre los casos de fraude y los casos normales, implementando tecnicas de machine learning.



## Preprocesamiento


1- **Manejo de nans**: a continuacion se muestra las columnas con valores nan y su porcentaje de nans.

* merch_zipcode : 15.11%

2- **Codificacion**: las columnas categoritcas son las siguientes.

```
['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
```

Se eliminara la columna **trans_num** directamente.

Para la columna **trans_date_trans_time** se hara una conversion numerica para cada fecha.

En cuanto al resto de variables categoricas, tenemos el problema de que cada una de ellas tiene muchos valores diferentes,
por tanto, si implementamos OneHotEncoding como hacemos comunmente, el dataframe tomara un peso ingente. Por lo mismo, utilizaremos
*Frecuency Encoding*, una forma de codificacion en la cual se reemplaza cada categoria por su frecuencia de aparicion en el subconjunto.


Las variables categoricas restantes son:

```
['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob']
```

3- **Scalers:** se aplico *RobustScaler*.

4- **Sampling Bias**: no se va a estudiar.

5- **Estudio de correlaciones**: teniendo en cuenta que, el proceso de extraccion y seleccion logro reducir a 1 caracteristica, no fue necesario estudiar correlaciones.

6- **Extraccion y/o seleccion de caracteristicas**: despues de realizar el estudio de seleccion de caracteristicas, se concluyo que las mas relevantes son las siguientes.



```

                  Feature  Importance
4                     amt    0.379329
3                category    0.097641
2                merchant    0.061818
0   trans_date_trans_time    0.050806
17              unix_time    0.050269
16                    dob    0.048864
8                  street    0.045294
18              merch_lat    0.035088
19             merch_long    0.034913
9                    city    0.031160
20          merch_zipcode    0.028980
14               city_pop    0.018581
15                    job    0.018402
6                    last    0.017819
5                   first    0.017096
1                  cc_num    0.013587
12                    lat    0.012620
13                   long    0.011645
11                    zip    0.011503
```
Se aplico PCA exitosamente, logrando reducir a una sola caracteristica, manteniendo 0.999% del ratio de varianza.

7- **Estudio de distribucion gaussiana de las features**: fue imposible dada las dimensiones del dataset

Nota: nos enfretamos a un problema bastante molesto y es que, al utizar el pipeline que se presenta a continuacion:

```
    pipeline = Pipeline([
        ("date_converter",  DateConverter("trans_date_trans_time")),
        ("imputer",         CustomImputer(strategy="most_frequent")),
        ("encoding",        FrecuencyEncoding()),
        ("scaler",          CustomScaler(df.drop(target, axis=1).columns.tolist())),
        ("pca",             PCA(n_components=0.999))
    ])


```

El algoritmo de PCA *genera un dataframe completamente nuevo cuyos indices no se corresponden con los de los dataframes originales*, esto provocaba que, al unir los dataframes producto de las transformaciones con las columnas target, se generaran valores **NaN**.

Para solucionar esto, se especifico el parametro index al generar el dataframe producto de la transformacion.

```
  pipeline.fit(df_train.drop(target, axis=1), df_train[target])
    X_train = pd.DataFrame(pipeline.transform(df_train.drop(target, axis=1)),   index=df_train.index)
    X_test  = pd.DataFrame(pipeline.transform(df_test.drop(target, axis=1)),    index=df_test.index)
    X_val   = pd.DataFrame(pipeline.transform(df_val.drop(target, axis=1)),     index=df_val.index)

```

Ademas, en este proceso de preprocesamiento aprendimos las implicaciones que tiene el uso del metodo **fit**, y aprendimos la forma correcta de dividir el conjunto ed datos y de transformarlo posteriormente.

El codigo utilizado finalmente para preprocesar fue el siguiente.

```
from preprocess.date_converter import DateConverter
from sklearn.pipeline import Pipeline
from preprocess.nan_fixer import  CustomImputer
import pandas as pd
from preprocess.encoding import FrecuencyEncoding
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from preprocess.scaler import CustomScaler



def basic_preprocess(df, target : str):

    important_features = ['amt', 'category', 'merchant', 'trans_date_trans_time', 'unix_time', 'dob', 'street', 'merch_lat', 'merch_long', 'city', 'merch_zipcode', 'city_pop', 'job', 'last', 'first', 'cc_num', 'long', 'zip', "is_fraud"]

    df = df.drop(["trans_num", "Unnamed: 0"], axis=1)
    df = df[important_features]
    df = df.copy()

    df_train, unseen_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df[target])
    df_val, df_test = train_test_split(unseen_df, test_size=0.5, shuffle=True, random_state=42, stratify=unseen_df[target])

    pipeline = Pipeline([
        ("date_converter",  DateConverter("trans_date_trans_time")),
        ("imputer",         CustomImputer(strategy="most_frequent")),
        ("encoding",        FrecuencyEncoding()),
        ("scaler",          CustomScaler(df.drop(target, axis=1).columns.tolist())),
        ("pca",             PCA(n_components=0.999))
    ])


    pipeline.fit(df_train.drop(target, axis=1), df_train[target])
    X_train = pd.DataFrame(pipeline.transform(df_train.drop(target, axis=1)),   index=df_train.index)
    X_test  = pd.DataFrame(pipeline.transform(df_test.drop(target, axis=1)),    index=df_test.index)
    X_val   = pd.DataFrame(pipeline.transform(df_val.drop(target, axis=1)),     index=df_val.index)


    df_train    = pd.concat([X_train, df_train[target]], axis=1)
    df_test     = pd.concat([X_test, df_test[target]], axis=1)
    df_val      = pd.concat([X_val, df_val[target]], axis=1)
    

    return [df_train, df_test, df_val]

```

Correccion: despues de haber entrenado los algoritmos utilizando el procedimiento de preprocesamiento anterior, procedimos a quitar el paso del PCA del pipeline para estudiar los resultados de los algoritmos sin aplicar PCA, esto nos llevo a obtener resultados muy raros. Primero, el pipeline empezo a generar unas advertencias del siguiente estilo:

```
/home/santiago/.local/lib/python3.10/site-packages/sklearn/pipeline.py:62: FutureWarning: This Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments before using other methods such as transform, predict, etc. This will raise an error in 1.8 instead of the current warning.
warnings.warn(
/home/santiago/.local/lib/python3.10/site-packages/sklearn/pipeline.py:62: FutureWarning: This Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments before using other methods such as transform, predict, etc. This will raise an error in 1.8 instead of the current warning.
warnings.warn(
```

Ademas, el pipeline estaba resultando en multiples valores Nan despues de las transformaciones.

Despues de algo de tiempo de estudio nos dimos cuenta de lo siguiente:

1- Advertencias: las advertencias estaban siendo producto de un error de construccion en los transformadores. `scikit-learn` obliga a que todos los parametros recibidos en el constructor de un transformador sean atributos del mismo, y los que no se reciban, deben tener la estructura de `atributo_`. Revisar este [link](https://scikit-learn.org/stable/developers/develop.html#instantiation).

2- Los valores Nan: estos estaban siendo generados por el transformador `CustomImputer`. Como sabemos, dicho transformador esta construido encima de `SimpleImputer`, lo que hace es tomar la transformacion producida por `SimpleImputer` y convertir en dataframe el resultado. El problema es que, dicho resultado, se estaba transformando en un dataframe sin especificar indices ni columnas. Este termino siendo el transformador arreglado.

```
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class CustomImputer(BaseEstimator, TransformerMixin):
    """
        Implementa el SimpleImputer pero, a diferencia del mismo, retorna
        dataframes y no matrices de numpy
    """
    def __init__(self, strategy : str, attributes) -> None:
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=strategy)
        self.attributes = attributes
    def fit(self, X, y=None):
        print("CustomImputer Ajustado")
        self.imputer_.fit(X[self.attributes], y)
        return self

    def transform(self, X, y=None):
        transformed_data = self.imputer_.transform(X.copy()[self.attributes])
        transformed_data = pd.DataFrame(transformed_data, columns=self.attributes, index=X[self.attributes].index)
        X[self.attributes] = transformed_data
        print("CustomImputer transformado")
        return X

```

3- Ademas de lo anterior, habia un problema en la comunicacion entre `CustomImputer` y `DateConverter`. Esto pasaba por que, por alguna razon, el `CustomImputer` estaba transformando el tipo de las columnas que no convertia a `object`. Esto tambien fue solucionado.

4- Por ultimo, tambien nos dimos cuenta de que, la llamada al metodo `fit()` del Pipeline estaba generando un comportamiento inesperado, principalmente por que de por si, los pipelines estan pensados para ser construidos con un estimador al final. La version final de la funcion `basic_preprocess` es la siguiente:

```
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
        ("scaler",          CustomScaler(df.drop(target, axis=1).columns.tolist()))
    ])
    

    X_train = pd.DataFrame(pipeline.fit_transform(df_train.drop(target, axis=1), df_train[target]),   index=df_train.index)
    X_test  = pd.DataFrame(pipeline.transform(df_test.drop(target, axis=1)),    index=df_test.index)
    X_val   = pd.DataFrame(pipeline.transform(df_val.drop(target, axis=1)),     index=df_val.index)


    df_train    = pd.concat([X_train, df_train[target]], axis=1)
    df_test     = pd.concat([X_test, df_test[target]], axis=1)
    df_val      = pd.concat([X_val, df_val[target]], axis=1)

    
    return [df_train, df_test, df_val]

```

## Entrenamiento (previo a deteccion de problemas de preprocesamiento)

La idea sera evaluar los resultados de cada uno de los algoritmos utilizando y sin utilizar PCA, para revisar que tanto varian los resultados.

*Se llevara a cabo un proceso iterativo de seleccion de modelo, entrenamiento y evaluacion de cada algoritmo por separado*

Nota: al final se opto por no usar K-MEANS ni DBSCAN dado su peso.



### Entrenamiento utilizando PCA 


### Isolation Forest

En este caso, el resultado del proceso de seleccion de modelo fue el siguiente:

```
Mejores hiperparámetros: {'random_state': 42, 'n_estimators': 100, 'max_samples': 0.65, 'max_features': 0.75, 'contamination': 0.01, 'bootstrap': False}
Mejor precision: 0.05793688126031915

```

El codigo utilizado para realizar la seleccion de modelo fue:

```
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

```


### GMM

Fue interesante en este proceso aprender mas sobre este algoritmo y su forma de funcionamiento en la practica. Este algoritmo, una vez es entrenado, a la hora de predecir implementa el metodo `score_samples()`, el cual toma las entradas y retorna un arreglo con numeros que representan las probabilidades de cada ejemplo de ser *normales*. La cuestion es que, el algoritmo no tiene una forma automatizada de generar el umbral mas optimo, es decir, el porcentaje a partir del cual se empieza a considerar un ejemplo como normal o anomalo, es algo que se tiene que hacer manualmente como en el ejemplo de abajo.

Este algoritmo logro resultados bastante por debajo de las expectativas, logrando para su mejor umbral un f1_score para la clase positiva de 0.012%.


Se empleo el siguiente codigo para entrenamiento y evaluacion del algoritmo:


```
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
grid_search.fit(df_train.query(f"{TARGET} == 0").drop(TARGET, axis=1))
print(grid_search.best_score_)

joblib.dump(grid_search.best_estimator_, "gmm.joblib")

```


Su mejor combinacion de hiperparametros fue:


```
{'covariance_type': 'diag', 'init_params': 'kmeans', 'max_iter': 100, 'means_init': None, 'n_components': 2, 'n_init': 1, 'precisions_init': None, 'random_state': None, 'reg_covar': 1e-06, 'tol': 0.001, 'verbose': 0, 'verbose_interval': 10, 'warm_start': False, 'weights_init': None}

```

## Entrenamiento (posterior a mejora del proceso de preprocesamiento)


### Entrenamiento utilizando PCA 


### Isolation Forest

Se utilizo el mismo algoritmo para la seleccion de modelo que mas arriba.

La mejor combinacion de hiperparametros fue:


### GMM


Se utilizo el mismo algoritmo para la seleccion de modelo que mas arriba.

La mejor combinacion de hiperparametros fue:


### Entrenamiento sin PCA 


### Isolation Forest

Se utilizo el mismo algoritmo para la seleccion de modelo que mas arriba.

La mejor combinacion de hiperparametros fue:


### GMM


Se utilizo el mismo algoritmo para la seleccion de modelo que mas arriba.

La mejor combinacion de hiperparametros fue:
