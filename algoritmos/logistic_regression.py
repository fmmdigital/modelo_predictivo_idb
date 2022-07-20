#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
import pathlib
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import numpy as np
import datetime as dt
import json
import utils
import random
import openpyxl
from sklearn import metrics
import collections
import pickle

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

print("Preparando modelo logistic regression ")
inicio_time = dt.datetime.now()

#genero id random de modelo para vincular rtdos con feature importance
id_modelo = random.randint(100000, 900000)

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
print('Estimando modelo logistic regression')
if params['penalty'] ==  'l1':
   model = LogisticRegression(**params, solver='liblinear').fit(X_proc, Y)
else:
   model = LogisticRegression(**params, solver='lbfgs', max_iter = 1000).fit(X_proc, Y)

   
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model, X_proc, Y, scoring='roc_auc', cv=cv, n_jobs=-1)

# Save model
with open(f'results/modelsobjects/reglog_{modelo}_idexperimento{id_experimento}_modelo{id_modelo}.pkl', 'wb') as f:
    pickle.dump(model, f)


# Predict en test
print('Generando prediccion sobre test')
pred_y = model.predict(test_x)

# Results
print('Calculando metricas')
tn, fp, fn, tp,accuracy,balanced_accuracy,precision,recall,f1score,roc_auc_score,roc_auc_score_train = utils.generar_metricas_test(test_y,pred_y,scores)

r2 = metrics.r2_score(test_y, pred_y)


# Coeficients
coef_dict = {}
for coef, feat in zip(model.coef_[0],X.columns):
    coef_dict[feat] = coef
    
df_coef = pd.DataFrame.from_dict(coef_dict, orient = 'index')   
 
filepath = f'results/feature_importance/id_experimento_{id_experimento}_logreg_{id_modelo}.xlsx'
wb = openpyxl.Workbook()
wb.save(filepath)

with pd.ExcelWriter(f'results/feature_importance/id_experimento_{id_experimento}_logreg_{id_modelo}.xlsx', mode = 'a') as writer:  
    df_coef.to_excel(writer)

# Output results
print('Guardando resultados')
df_resultados_modelo = utils.generar_dataframe_resultados(id_experimento, 
                                                          inicio_time,
                                 modelo,
                                 'regresion_logistica',
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
                                 {'r2':r2},
                                 train_name_x,
                                 train_x.shape,
                                 train_name_y,
                                 train_y.shape,
                                 test_name_x,
                                 test_x.shape,
                                 test_name_y,
                                 test_y.shape,
                                 {'id_modelo':id_modelo})


df_resultados_modelo.to_csv('results/log.csv',sep = '|', index = False, mode = 'a', header = False)



