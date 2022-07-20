#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:47:01 2022

@author: mac
"""

#!/usr/bin/env python3
import pathlib
import pandas as pd
import sys
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn import metrics, preprocessing
import numpy as np
import datetime as dt
import json
import utils #import algoritmos.utils as utils
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


print("Preparando modelo lightgbm ")

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
X = utils.reduce_mem_usage(X)
train_name_y = f'data/{modelo}/train/{dataset}_y_train.csv' # desarrollo
train_y = pd.read_csv(train_name_y)
Y = train_y.values.ravel()
import re
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X['ruc_entidad'] = X['ruc_entidad'].astype('category')

# test
test_name_x = f'data/{modelo}/test/{dataset}_x_test.csv' # desarrollo
test_x = pd.read_csv(test_name_x, index_col=0)
test_x = test_x.loc[:, ~test_x.columns.isin(['codigo_contrato'])]
test_x = test_x.select_dtypes(include=numerics)
test_x = utils.reduce_mem_usage(test_x)
test_name_y = f'data/{modelo}/test/{dataset}_y_test.csv' # desarrollo
test_y = pd.read_csv(test_name_y)
test_x = test_x.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

test_x['ruc_entidad'] = test_x['ruc_entidad'].astype('category')

# Resampling
print(f'Cantidad de 1 despues de resampling: {collections.Counter(Y)}')
X,Y = utils.aplicar_resampling(resampling, X, Y)
print(f'Cantidad de 1 despues de resampling: {collections.Counter(Y)}')

# Standarization
X_proc,test_x = utils.preprocessing_standarization(scaling, X,test_x)

# Model

print('Estimando modelo light gbm')
model = lgb.LGBMClassifier(**params).fit(X_proc, Y)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X_proc, Y, scoring='roc_auc', cv=cv, n_jobs=-1)

# Save model
model.booster_.save_model(f'results/modelsobjects/lgbm_{modelo}_idexperimento{id_experimento}_modelo{id_modelo}.txt')

# Predict en test
print('Generando prediccion sobre test')
pred_y = model.predict(test_x)
pred_proba_y = model.predict_proba(test_x)

# Envio prediccion a csv

test_y = test_y['Y_sobretiempos'].to_numpy()
df_export = pd.DataFrame()      
df_export['Y_true'] = pd.Series(test_y)
df_export['Y_pred'] = pd.Series(pred_y)
df_export['Y_pred_proba'] = pd.Series(pred_proba_y[:,1])

df_export.to_csv(f'results/id_experimento_{id_experimento}_lightgbm_{id_modelo}_prediccion_test.csv', index = False, sep = '|', float_format='%.4f')


# Results
print('Calculando metricas')
tn, fp, fn, tp,accuracy,balanced_accuracy,precision,recall,f1score,roc_auc_score,roc_auc_score_train = utils.generar_metricas_test(test_y,pred_y,scores)
print(f'roc_auc_score = {roc_auc_score}')
print(f'f1 = {f1score}')
print(f'fp = {fp} - tp = {tp}')

# Feature importance
importances = model.feature_importances_
important_features = pd.Series(data=model.feature_importances_,index=X.columns)
important_features.sort_values(ascending=False,inplace=True)
 
filepath = f'results/feature_importance/id_experimento_{id_experimento}_lightgbm_{id_modelo}.xlsx'
wb = openpyxl.Workbook()
wb.save(filepath)

with pd.ExcelWriter(f'results/feature_importance/id_experimento_{id_experimento}_lightgbm_{id_modelo}.xlsx', mode = 'a') as writer:  
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
                                 'lightgbm',
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