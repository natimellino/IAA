import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import pickle
import sys
from sklearn import datasets
from sklearn.metrics import accuracy_score
import ej4tp1


from ej4tp1 import generate_dataframe_a, generate_dataframe_b

warnings.filterwarnings("ignore")

def dim_ann():
    ds = [2, 4]

    test_errors_a = []
    test_errors_b = []
    train_errors_a = []
    train_errors_b = []

    dd = []
    for d in ds:
        dd.append(d)
        cols = list(map(str, list(range(0, d))))
        # Generamos el conjunto de testeo
        test_a = generate_dataframe_a(d = d, n = 10000, C = 0.78)
        test_set_a = test_a.loc[ : , cols ]
        y_test_set_a = test_a.loc[:, 'Class']

        test_b = generate_dataframe_b(d = d, n = 10000, C = 0.78)
        test_set_b = test_b.loc[ : , cols ]
        y_test_set_b = test_b.loc[:, 'Class']
        # Generamos el conjunto de entrenamiento
        train_set_a = generate_dataframe_a(d = d, n = 250, C = 0.78)
        train_set_b = generate_dataframe_b(d = d, n = 250, C = 0.78)

        # Creamos el árbol y lo entrenamos
        Xtrain_a, ytrain_a = train_set_a.loc[ : , cols ], train_set_a.loc[:, 'Class']

        clf_a = pickle.load(open(f'ann/diag-{d}.plk', 'rb'))
        
        Xtrain_b, ytrain_b = train_set_b.loc[ : , cols ], train_set_b.loc[:, 'Class']
        clf_b = pickle.load(open(f'ann/paral-{d}.plk', 'rb'))

        # Predecimos sobre el conjunto de testeo
        prediction_a = clf_a.predict(test_set_a)
        prediction_b = clf_b.predict(test_set_b)
        prediction_train_a = clf_a.predict(train_set_a.loc[ : , cols])
        prediction_train_b = clf_b.predict(train_set_b.loc[ : , cols])

        # Guardamos el error (1 - accuracy) sobre el conjunto de testeo
        # y el de entrenamiento.
        test_errors_a.append(1 - accuracy_score(y_test_set_a, prediction_a))
        test_errors_b.append(1 - accuracy_score(y_test_set_b, prediction_b))
        train_errors_a.append(1 - accuracy_score(train_set_a.loc[:, 'Class'], prediction_train_a))
        train_errors_b.append(1 - accuracy_score(train_set_b.loc[:, 'Class'], prediction_train_b))

    # Creamos los dataframes con cada valor de C y su respectivo error (20 errores por cada N habrá)
    df_a = pd.DataFrame({})
    df_a['d'] = dd
    df_a['Test Error'] = test_errors_a
    df_a['Train Error'] = train_errors_a

    df_b = pd.DataFrame({})
    df_b['d'] = dd
    df_b['Test Error'] = test_errors_b
    df_b['Train Error'] = train_errors_b

    # Agrupamos y calculamos el promedio de error para cada valor de C
    mean_test_error_a = df_a.groupby('d')['Test Error'].mean().to_numpy()
    mean_test_error_b = df_b.groupby('d')['Test Error'].mean().to_numpy()
    mean_train_error_a = df_a.groupby('d')['Train Error'].mean().to_numpy()
    mean_train_error_b = df_b.groupby('d')['Train Error'].mean().to_numpy()
    # Guardamos los resultados obtenidos en los dataframes
    error_df_a = pd.DataFrame({})
    error_df_a['d'] = ds
    error_df_a['Test Error'] = mean_test_error_a
    error_df_a['Train Error'] = mean_train_error_a

    error_df_b = pd.DataFrame({})
    error_df_b['d'] = ds
    error_df_b['Test Error'] = mean_test_error_b
    error_df_b['Train Error'] = mean_train_error_b

    return (error_df_a, error_df_b)