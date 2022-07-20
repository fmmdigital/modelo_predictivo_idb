#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
import pandas as pd

#pathlib.Path(__file__).parent.resolve() # para correr desde terminal

# Importo dataframe
df = pd.read_csv('data/dataframe_v4_sin_ohe.csv')

# Reviso cantidad de nulls
def check_cantidad_nulls(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return(df[columns_with_nan].isna().sum())
check_cantidad_nulls(df)

# Dataframe que reemplaza nulls por ceros
df_ceros = df.fillna(0)
check_cantidad_nulls(df_ceros)
df_ceros.to_csv('data/dataframe_ceros_v4_sin_ohe.csv', index = False)

# Dataframe que reemplaza nulls por -9999
df_nuevenueve = df.fillna(-9999)
check_cantidad_nulls(df_nuevenueve)
df_nuevenueve.to_csv('data/dataframe_nuevenueve_v4_sin_ohe.csv', index = False)

# Dataframe que quita nulls
df_sinnulls = df.dropna(axis = 0)
check_cantidad_nulls(df_sinnulls)
df_sinnulls.to_csv('data/dataframe_sinnulls_v4_sin_ohe.csv', index = False)


