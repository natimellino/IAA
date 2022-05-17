import random
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from math import sqrt
from typing import List

def generate_dataframe(d: int, n: int, deviation: float, center0: List[int], center1: List[int]):    
    m = [deviation**2] * d
    # Generamos los n/2 puntos de la clase 0
    data0 = np.random.multivariate_normal(center0, np.diag(m), n//2)  
    # Generamos los n/2 puntos de la clase 1
    data1 = np.random.multivariate_normal(center1, np.diag(m), math.ceil(n/2))  


    # Concatenamos ambos arrays, primero colocamos los puntos correspondientes a 
    # la clase 0 y luego los correspondientes a la clase 1 así luego no es más
    # fácil clasificarlos.
    data = np.concatenate((data0, data1), axis=0)

    # Generamos las columnas que representarán las coordenadas de cada punto.
    cols = list(map(str, list(range(d))))
    # Generamos el dataframe
    df = pd.DataFrame(data, columns=cols)
    # Agregamos al final del dataframe la columna que representa la clase de
    # cada punto.
    df['Class'] = ([0] * (n // 2)) + ([1] * (math.ceil(n/2)))
    
    return df

def generate_dataframe_a(d: int, n: int, C: float): 
    deviation = C * sqrt(d)
    c0 = [-1] * d
    c1 = [1] * d
    return generate_dataframe(d, n, deviation, c0, c1)

def generate_dataframe_b(d: int, n: int, C: float):
    deviation = C
    c0 = [-1] + ([0] * (d - 1))
    c1 = [1] + ([0] * (d - 1))
    return generate_dataframe(d, n, deviation, c0, c1)

def ej4tp1():
    ds = [2, 4, 6, 8, 16, 32]

    test_errors_a = []
    test_errors_b = []
    train_errors_a = []
    train_errors_b = []

    dd = []
    for d in ds:
        for i in range(0, 20):
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

            clf_a = DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=0.005,random_state=0,min_samples_leaf=5)
            clf_a.fit(Xtrain_a, ytrain_a)

            Xtrain_b, ytrain_b = train_set_b.loc[ : , cols ], train_set_b.loc[:, 'Class']
            clf_b = DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=0.005,random_state=0,min_samples_leaf=5)
            clf_b.fit(Xtrain_b, ytrain_b)

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

    # Graficamos
    # plt.plot(error_df_a['d'], error_df_a['Test Error'], 'r')
    # plt.plot(error_df_b['d'], error_df_b['Test Error'], 'b')
    # plt.plot(error_df_a['d'], error_df_a['Train Error'], 'g')
    # plt.plot(error_df_b['d'], error_df_b['Train Error'], 'y')
    # plt.legend(['Diag Test', 'Paral Test', 'Diag Train', 'Paral Train'], bbox_to_anchor=(0.75, 1.15), ncol=2)
    # plt.xlabel('d')
    # plt.ylabel('Error')