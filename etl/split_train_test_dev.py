#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from  sklearn.model_selection import train_test_split
import pandas as pd
import pathlib

#pathlib.Path(__file__).parent.resolve() # para correr desde terminal

# Check proporcion de positive labels en cada grupo
def check_positive_cases(y_split):
    y_np = y_split.to_numpy()
    num_zeros = (y_np == 0).sum()
    num_ones = (y_np == 1).sum()
    return(num_ones / (num_zeros + num_ones))

# Guardar csv
def save(path,x,y,x_str,y_str):
    x.to_csv(path.joinpath(f'{file.stem}_{x_str}.csv'),index = False)
    y.to_csv(path.joinpath(f'{file.stem}_{y_str}.csv'), index = False)
    return()



# Listo variables Y
ys = ['Y_sobrecostos_suma','Y_sobretiempos']
y_dir = ['sobrecostos','sobretiempos']

# dataframe para guardar caracteristicas de cada dataset
df_resumen = pd.DataFrame(columns = ['modelo','nombre_dataset','split','x_shape','y_shape','proporcion_casos_positivos'])

for file in pathlib.Path('data').glob("*.csv"):
    # DataFrame
    df = pd.read_csv(file)
    
    for nombre_y,y_d in zip(ys,y_dir):
        # X Y
        y_labels = df[[nombre_y]]
        x_data = df.loc[:, ~df.columns.isin(['Unnamed: 0','Y_sobrecostos', 'Y_sobrecostos_suma',
           'Y_sobretiempos', 'codigoconvocatoria'])]
        
        # Split
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.2, stratify = y_labels, random_state = 42)
        
        # Check size
        print(f'X train shape = {x_train.shape}')
        print(f'X test shape = {x_test.shape}')
        
        # Check proporcion de 1
        print(f'Y train proporcion de casos positivos = {check_positive_cases(y_train)}')
        print(f'Y test proporcion de casos positivos = {check_positive_cases(y_test)}')
        
        # Guardo dataset en cada carpeta
        train_path = pathlib.Path('data').joinpath(f'{y_d}','train')
        test_path = pathlib.Path('data').joinpath(f'{y_d}','test')
        
        save(train_path,x_train,y_train,'x_train','y_train')
        save(test_path,x_test,y_test,'x_test','y_test')
        
        # Guardo datos de split
        df_resumen = df_resumen.append(pd.Series([nombre_y,file.stem,'train',x_train.shape,y_train.shape,check_positive_cases(y_train)], 
                  index = df_resumen.columns), ignore_index = True)
        df_resumen = df_resumen.append(pd.Series([nombre_y,file.stem,'test',x_test.shape,y_test.shape,check_positive_cases(y_test)], 
                  index = df_resumen.columns), ignore_index = True)
    
df_resumen.to_excel('data/resumen_dataframes_sobretiempos.xlsx', index = False)
        
    
    

    
