# Anomalies

El objetivo de este proyecto sera poner en practica nuestros conocimientos recientes de aprendizaje no supervisado y semi-supervisado tomando un dataset desequilibrado (producto del estudio de anomalias), y revisar cual de los siguientes algoritmos genera mejores resultados:

* K-MEANS
* DB-SCAN
* Isolation Forest
* GMM

El [dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) que trataremos contiene informacion acerca de transacciones bancarias, de las cuales, el 0,57% son fraude. El objetivo sera tratar de identificar relaciones entre los casos de fraude y los casos normales, implementando tecnicas de machine learning.



## Preprocesamiento


1- Manejo de nans: a continuacion se muestra las columnas con valores nan y su porcentaje de nans.

**merch_zipcodejj** : 15.11%

2- Codificacion: las columnas categoritcas son las siguientes.

```
['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
```

Se eliminara la columna **trans_num** directamente.

Para la columna **trans_date_trans_time** se hara una conversion numerica para cada fecha.

En cuanto al resto de variables categoricas, tenemos el problema de que cada una de ellas tiene muchos valores diferentes,
por tanto, si implementamos OneHotEncoding como hacemos comunmente, el dataframe tomara un peso ingente. Por lo mismo, utilizaremos
*Frecuency Encoding*, una forma de codificacion en la cual se reemplaza cada categoria por su frecuencia de aparicion en el subconjunto.
La implementacion de frecuency Encoding conllevaria tener mucho cuidado con el manejo de los subcojuntos de entrenamiento y test, ya que la
frecuencia de cada categoria debe ser relativa a su aparicion en el subconjunto.

Las variables categoricas restantes son:

```
['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob']
```

3- Scalers: se aplico *RobustScaler*.

4- Sampling Bias: no se va a estudiar.

5- Estudio de correlaciones.

6- Extraccion y/o seleccion de caracteristicas: despues de realizar el estudio de seleccion de caracteristicas, se concluyo que las mas relevantes son las siguientes.

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

7- Estudio de distribucion gaussiana de las features: fue imposible dada las dimensiones del dataset

Nota: nos enfretamos a un problema bastante molesto y es que, al utizar el pipeline que se presenta a continuacion:

```
        pipeline = Pipeline([
            ("date_converter",  DateConverter("trans_date_trans_time")),
            ("imputer", FixNanRowsWithMean()),
            ("encoding", FrecuencyEncoding()),
            ("scaler", CustomScaler(df.drop(target, axis=1).columns.tolist())),
            ("pca", PCA(n_components=0.99))
        ])

```

El algoritmo de PCA genera un dataframe completamente nuevo cuyos indices no se corresponden con los de los dataframes, esto provocaba que, al unir los dataframes producto de las transformaciones con las columnas target, se generaran valores nan.

Para solucionar esto, se especifico el parametro index al generar el dataframe producto de la transformacion.

```
X_train = pd.DataFrame(pipeline.fit_transform(df_train.drop(target, axis=1), df_train[target]), index=df_train.index)
    X_test = pd.DataFrame(pipeline.transform(df_test.drop(target, axis=1)), index=df_test.index)
```

Ademas, en este proceso de Preprocesamiento aprendimos las implicaciones que tiene el uso del metodo **fit**, y aprendimos la forma correcta de dividir el conjunto ed datos y de transformarlo posteriormente.

## Entrenamiento

*Se llevara a cabo un proceso iterativo de seleccion de modelo, entrenamiento y evaluacion de cada algoritmo por separado*

### K-MEANS

### DB-SCAN

### Isolation Forest

### GMM
