#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:51:17 2022

@author: mac
"""

#!/usr/bin/env python3
import pathlib
import pandas as pd
import sys
from sklearn.linear_model  import SGDClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn import metrics, preprocessing
import datetime as dt
import json
import random
import openpyxl
import utils
import time
import collections

#pathlib.Path(__file__).parent.resolve() # para correr desde terminal

# Leo variables entorno
modelo = sys.argv[1] #produccion
dataset = sys.argv[2] #produccion
combo_hiperparametros = sys.argv[3] # produccion
id_experimento = sys.argv[4] #produccion
scaling = sys.argv[5] #scaling
resampling = sys.argv[6] #resampling

combo_hiperparametros = combo_hiperparametros.replace('#',',')
combo_hiperparametros = combo_hiperparametros.replace('-','"')
combo_hiperparametros = json.loads(combo_hiperparametros)
params=combo_hiperparametros


print("Preparando modelo SGD ")

inicio_time = dt.datetime.now()


print(f"id_experimento = {id_experimento}, modelo = {modelo}, dataset = {dataset}, hiperparametros = {params}")

# train
train_name_x = f'data/{modelo}/train/{dataset}_x_train.csv' # desarrollo
train_x = pd.read_csv(train_name_x, index_col=0)
X = train_x.loc[:, ~train_x.columns.isin(['codigo_contrato'])]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X = X.select_dtypes(include=numerics)
train_name_y = f'data/{modelo}/train/{dataset}_y_train.csv' # desarrollo
train_y = pd.read_csv(train_name_y)
Y = train_y.values.ravel()

# test
test_name_x = f'data/{modelo}/test/{dataset}_x_test.csv' # desarrollo
test_x = pd.read_csv(test_name_x, index_col=0)
test_x = test_x.loc[:, ~test_x.columns.isin(['codigo_contrato'])]
test_x = test_x.select_dtypes(include=numerics)
test_name_y = f'data/{modelo}/test/{dataset}_y_test.csv' # desarrollo
test_y = pd.read_csv(test_name_y)


# Resampling
print(f'Cantidad de 1 despues de resampling: {collections.Counter(Y)}')
X,Y = utils.aplicar_resampling(resampling, X, Y)
print(f'Cantidad de 1 despues de resampling: {collections.Counter(Y)}')

# Standarization
X_proc,test_x = utils.preprocessing_standarization(scaling, X,test_x)
                                
# Model
print('Estimando modelo SGD')
model = SGDClassifier(**params).fit(X_proc, Y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X_proc, Y, scoring='roc_auc', cv=cv, n_jobs=-1)

# Predict en test
print('Generando prediccion sobre test')
pred_y = model.predict(test_x)

# Results
print('Calculando metricas')
tn, fp, fn, tp,accuracy,balanced_accuracy,precision,recall,f1score,roc_auc_score,roc_auc_score_train = utils.generar_metricas_test(test_y,pred_y,scores)


# Output results
print('Guardando resultados')

df_resultados_modelo = utils.generar_dataframe_resultados(id_experimento, 
                                                          inicio_time,
                                 modelo,
                                 'SGD',
                                 params,
                                 scaling,
                                 resampling + str((collections.Counter(Y))),
                                 tn, fp, fn, tp,
                                 accuracy,
                                 balanced_accuracy,
                                 precision,
                                 recall,
                                 f1score,
                                 roc_auc_score,
                                 roc_auc_score_train,
                                 None,
                                 train_name_x,
                                 train_x.shape,
                                 train_name_y,
                                 train_y.shape,
                                 test_name_x,
                                 test_x.shape,
                                 test_name_y,
                                 test_y.shape,
                                 None)


df_resultados_modelo.to_csv('results/log.csv',sep = '|', index = False, mode = 'a', header = False)
