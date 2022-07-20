#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:01:20 2022

@author: mac
"""

#!/usr/bin/env python3
import pathlib
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn import metrics, preprocessing
import numpy as np
import datetime as dt
import json
import utils
import datetime as dt
import random
import openpyxl
import collections

#pathlib.Path(__file__).parent.resolve() # para correr desde terminal

# Importo dataframes
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


#modelo = 'sobrecostos'
#dataset = dataset
#id_experimento = 17
#scaling = scaling

print("Preparando modelo random forest ")

#genero id random de modelo para vincular rtdos con feature importance
id_modelo = random.randint(100000, 900000)

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
print('Estimando modelo random forest')
model = RandomForestClassifier(**params).fit(X_proc, Y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X_proc, Y, scoring='roc_auc', cv=cv, n_jobs=-1)

# Predict en test
print('Generando prediccion sobre test')
pred_y = model.predict(test_x)

# Results
print('Calculando metricas')
tn, fp, fn, tp,accuracy,balanced_accuracy,precision,recall,f1score,roc_auc_score,roc_auc_score_train = utils.generar_metricas_test(test_y,pred_y,scores)


# Feature importance
importances = model.feature_importances_
important_features = pd.Series(data=model.feature_importances_,index=X.columns)
important_features.sort_values(ascending=False,inplace=True)
 
filepath = f'results/feature_importance/id_experimento_{id_experimento}_rf_{id_modelo}.xlsx'
wb = openpyxl.Workbook()
wb.save(filepath)

with pd.ExcelWriter(f'results/feature_importance/id_experimento_{id_experimento}_rf_{id_modelo}.xlsx', mode = 'a') as writer:  
    important_features.to_excel(writer)

# Save dataframe with Y and predicction
'''
test_name_x = f'data/{modelo}/test/{dataset}_x_test.csv' # desarrollo
test_x = pd.read_csv(test_name_x, index_col=0)
#test_x = test_x.loc[:, ~test_x.columns.isin(['codigo_contrato'])]
test_x = test_x.select_dtypes(include=numerics)
df_pred = pd.DataFrame(pred_y).rename(columns = {0 : 'Y_pred'})
test_x.reset_index(drop=True, inplace=True)
test_y.reset_index(drop=True, inplace=True)
df_pred.reset_index(drop=True, inplace=True)
df_y_pred = pd.concat( [test_x, test_y, df_pred], axis=1) 
df_y_pred.to_csv(f'results/analisis_error/prediccion_test_rf_{id_modelo}.csv', index = False)'''



# Output results
print('Guardando resultados')
df_resultados_modelo = utils.generar_dataframe_resultados(id_experimento, 
                                                          inicio_time,
                                 modelo,
                                 'Random_Forest',
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
                                 id_modelo)


df_resultados_modelo.to_csv('results/log.csv',sep = '|', index = False, mode = 'a', header = False)
