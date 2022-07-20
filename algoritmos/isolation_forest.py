#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import pandas as pd
import sys
from sklearn.ensemble import IsolationForest

pathlib.Path(__file__).parent.resolve() # para correr desde terminal

# Importo dataframe
# df_name = sys.argv[1] #produccion
df_name = 'data/dataframe.csv' # desarrollo
df = pd.read_csv(df_name)

X = df['monto_contratado_total']

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model_fit = model.fit(df[['monto_contratado_total','codigoconvocatoria']])
model_fit.predict(df[['monto_contratado_total','codigoconvocatoria']])

print(f'{df.shape}')