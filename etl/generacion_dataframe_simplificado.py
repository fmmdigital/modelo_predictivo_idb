#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
# leo dataframes
x_train_toy = pd.read_csv('data/sobrecostos/train/dataframe_sinnulls_x_train.csv')  
y_train_toy = pd.read_csv('data/sobrecostos/train/dataframe_sinnulls_y_train.csv')   
x_test_toy = pd.read_csv('data/sobrecostos/test/dataframe_sinnulls_x_test.csv')   
y_test_toy = pd.read_csv('data/sobrecostos/test/dataframe_sinnulls_y_test.csv')    


x = x_train_toy.loc[:, x_train_toy.columns.str.contains('ruc_entidad')]
len(x.columns)

x = x_train_toy.loc[:, x_train_toy.columns.str.contains('item_departamento_')]
len(x.columns)

x = x_train_toy.loc[:, x_train_toy.columns.str.contains('entidad_departamento_')]
len(x.columns)

x = x_train_toy.loc[:, x_train_toy.columns.str.contains('codigogrupo')]
len(x.columns)

 ############# Solo todos los onehotencoding  ################
 
# saco one-hot encoding - train
x_train_simple = x_train_toy.loc[:, ~x_train_toy.columns.str.contains('ruc_entidad')]
x_train_simple = x_train_simple.loc[:, ~x_train_simple.columns.str.contains('item_departamento_')]
x_train_simple = x_train_simple.loc[:, ~x_train_simple.columns.str.contains('entidad_departamento_')]
x_train_simple = x_train_simple.loc[:, ~x_train_simple.columns.str.contains('grupo_familia')]
x_train_simple = x_train_simple.loc[:, ~x_train_simple.columns.str.contains('codigogrupo')]

x_train_simple.to_csv('data/sobrecostos/train/dataframe_sinnulls_simple_x_train.csv', index = False)
y_train_toy.to_csv('data/sobrecostos/train/dataframe_sinnulls_simple_y_train.csv', index = False)

# saco one-hot encoding - test
x_test_simple = x_test_toy.loc[:, ~x_test_toy.columns.str.contains('ruc_entidad')]
x_test_simple = x_test_simple.loc[:, ~x_test_simple.columns.str.contains('item_departamento_')]
x_test_simple = x_test_simple.loc[:, ~x_test_simple.columns.str.contains('entidad_departamento_')]
x_test_simple = x_test_simple.loc[:, ~x_test_simple.columns.str.contains('grupo_familia')]
x_test_simple = x_test_simple.loc[:, ~x_test_simple.columns.str.contains('codigogrupo')]

x_test_simple.to_csv('data/sobrecostos/test/dataframe_sinnulls_simple_x_test.csv', index = False)
y_test_toy.to_csv('data/sobrecostos/test/dataframe_sinnulls_simple_y_test.csv', index = False)

 ############# Solo saco onehotencoding problematico ################
 
# saco solo one-hot encoding  problematicos- train
x_train_simple = x_train_toy.loc[:, ~x_train_toy.columns.str.contains('ruc_entidad')]
x_train_simple = x_train_simple.loc[:, ~x_train_simple.columns.str.contains('grupo_familia')]

x_train_simple.to_csv('data/sobrecostos/train/dataframe_sinnulls_simple_ohe_x_train.csv', index = False)
y_train_toy.to_csv('data/sobrecostos/train/dataframe_sinnulls_simple_ohe_y_train.csv', index = False)

# saco one-hot encoding - test
x_test_simple = x_test_toy.loc[:, ~x_test_toy.columns.str.contains('ruc_entidad')]
x_test_simple = x_test_simple.loc[:, ~x_test_simple.columns.str.contains('grupo_familia')]

x_test_simple.to_csv('data/sobrecostos/test/dataframe_sinnulls_simple_ohe_x_test.csv', index = False)
y_test_toy.to_csv('data/sobrecostos/test/dataframe_sinnulls_simple_ohe_y_test.csv', index = False)


