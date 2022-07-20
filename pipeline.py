import sys
import yaml
import itertools as it
import os
import pandas as pd
import json
import time

#pathlib.Path(__file__).parent.resolve() # para correr desde terminal

# Experiment
#experiment_file = sys.arvs[0]
with open('experiments/sobretiempos/experimento_40.yaml', 'r') as file:
    experimento_file = yaml.safe_load(file)
    
id_experimento = experimento_file['experimento']['id_experimento']
modelo = experimento_file['experimento']['modelo']

lista_algoritmos = list(experimento_file.keys())
lista_algoritmos.pop(0) #quito caracteristicas del experimento

for algoritmo in lista_algoritmos:
    # varios
    script = experimento_file[algoritmo]["script"]
    datasets = experimento_file[algoritmo]["datasets"]
    stanzarization = experimento_file[algoritmo]["scaling"]
    resampling = experimento_file[algoritmo]["resampling"]
    
    # Genero combinacion de hiperparametros a probar
    hiperparametros = experimento_file[algoritmo]["hiperparametros"]
    all_hyper = list(hiperparametros.keys())
    combinations = it.product(*(hiperparametros[hyper] for hyper in all_hyper))
    todas_combinatorias = [dict(zip(all_hyper, comb)) for comb in list(combinations)]
    
    # Aplico script de algoritmo a cada dataset
    for dataset in datasets:
        print(dataset)
        for scaling in stanzarization: 
            print(scaling)
            for sampling in resampling: 
                print(sampling)
                for combo_hiperparametros in todas_combinatorias: 
                    print(combo_hiperparametros)
                    # llamo al script del algoritmo
                    combo_hiperparametros = json.dumps(combo_hiperparametros).replace(" ", "") #saco espacios en blanco
                    combo_hiperparametros = combo_hiperparametros.replace(',','#') # saco comas
                    combo_hiperparametros = combo_hiperparametros.replace("\"",'-') # saco comillas
                    os.system(f"{script} {modelo} {dataset} {combo_hiperparametros} {id_experimento} {scaling} {sampling}")
        

            

    