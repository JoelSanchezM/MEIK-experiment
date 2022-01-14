#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:53:28 2020

@author: joel
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold,
    cross_val_score,
)
from sklearn.preprocessing import MinMaxScaler
import os
import json
from metaheuristicas import PSOOptimizer
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import minmax_scale

def cost(x, *args):
    """
    Funcion a minimizar: Esta funcion hace kfold y regresa el promedio de
    los resultados. Utiliza root mean mean squared error
    """
    svr__C = x[0]
    svr__gamma = x[1]
    svr__epsilon = x[2]
    #svr__degree = x[3]
    params = {
                'C': svr__C,
                'gamma': svr__gamma,
                'epsilon': svr__epsilon,
    #            'degree': svr__degree
    }
    keyargs = args[0]['args']
    estimator = keyargs['Estimator']
    estimator.set_params(**params)
    cv = keyargs['CV']
    X = keyargs['X']
    y = keyargs['y']
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    scores = np.asarray(scores)


    return scores.mean()




def importar_datasets(nombre: str):
    """
    Toma el nombre de un archivo y lo regresa como dataset;
    debe tener el formato de libsvm donde las primeras columnas
    son las características y la última es el valor a predecir
    o clase.
    """
    data = np.loadtxt(nombre, skiprows=1, delimiter=",")
    x_data = data[..., :-1]
    y_data = data[..., -1]

    # Escalar los datos entre -1 y 1
    escalar = MinMaxScaler(feature_range=(-1, 1))
    x_data = escalar.fit_transform(x_data, y=y_data)

    return x_data, y_data

def importar_meik(nombre: str):
    df = pd.read_csv(nombre)
    iovars    = ['EDAD','AF','AHF CA','MENARCA','HIJO','RIESGO']


    validos = df[iovars]
    validos = validos.dropna()                  
    iovars.remove('RIESGO')
    X  = validos[iovars]
    columns = X.columns
    X = np.array(X)
    escalar = MinMaxScaler(feature_range=(-1,1))
    x_data = escalar.fit_transform(X)
    y = np.array(validos['RIESGO'])

    return x_data, y, escalar

initial = time()

nombre_archivo = "experimentomeik_trainartificialtesttesis.jsonl"


workspace = os.path.realpath("../datasets")
datasets = [
    'MEIK_artificial_eval.csv',
]

nombres_datasets = [workspace + "/" + n for n in datasets]
lims = {'C':[2**(-5),2**(8)],
        'Gamma':[2**(-5),2**(8)],
        'Epsilon':[0.1,0.9],
}

reps = 35 
per = reps*len(datasets)
per_c = 0

for n, d in zip(nombres_datasets, datasets):
   
    x_data, y_data, escalar = importar_meik(n)
    
    resultados = {}
    infocompleta = []
    for i in range(reps):
        for b in ['nearest']:
            df = pd.read_csv('../datasets/MEIK_limpio_comp.csv')
            x_test = df[['EDAD','AF','AHF CA','MENARCA','HIJO']]
            y_test = df['RIESGO TESIS']
           
            kfld = KFold(
                shuffle=True
            )
            
            args = {'Estimator': SVR(),
                    'CV':kfld,
                    'X':x_data,
                    'y':y_data}

            pso = PSOOptimizer(cost,lims,args=args)
            pso.optimize(30,30,b_handling=b)
            infocompleta.append(pso.history)
            params = pso.g_pos
            C = params[0]
            gamma = params[1]
            eps = params[2]
            svr = SVR(C=C, gamma=gamma,epsilon=eps)
            svr.fit(x_data,y_data)



            scores = cross_val_score(
                svr,
                x_data,
                y_data,
                scoring="neg_root_mean_squared_error",
            )


            x_test_scaled = escalar.transform(x_test)
            final_pred = svr.predict(x_test_scaled)
            final_mse = mean_squared_error(y_test, final_pred)
            final_rmse = np.sqrt(final_mse)
           
            scores *= -1.0

           
            resultados[d] = [
                scores.mean(),
                scores.std() * 2.0,
                {'C':params[0],'gamma':params[1],'eps':params[2]},
                np.min(scores),
                svr.n_support_[0] / len(y_data),
                svr.score(x_data,y_data),
                final_rmse,
                svr.score(x_test_scaled, y_test),
                pso.time_elapsed
            ]



            with open(nombre_archivo, "a") as f:
                 json.dump(resultados, f)
                 f.write("\n")
            per_c += 1
            print(per_c/per)

f=open('infotrainartificialtesttesis.txt','w')
for ele in infocompleta:
    print(str(ele), file=f)

f.close()
print(time()-initial)
