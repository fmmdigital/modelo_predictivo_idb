# Modelo Predictivo de sobrecostos y sobretiempos en compras públicas

Este repositorio corresponde al proyecto “Experimento con datos en compras y contrataciones gubernamentales Identificación predictiva de Riesgos” de
División de Gestión Fiscal del Banco Interamericano de Desarrollo (BID/FMM).

El repositorio contiene el pipeline y los scripts utilizados para entrenar un modelo de prediccion de sobrecostos y otro de sobretiempos en compras públicas utilizando distintos algoritmos de machine learning.

Para ejecutar el proceso de entrenamiento se debe definir el numero de experimento que se quiere llevar a cabo y ejecutar el script pipeline.py 

```
>python pipeline.py
```

El resultado de la ejecucion de pipeline.py genera archivos de modelos, feature importance y metricas en la carpeta 'results'.

## Contenido del repositorio
* algoritmos: contiene scripts de distintos algoritmos de machine learning
* etl: contiene scripts y jupyternotebook con los que se generaron los datasets para entrenar, las particiones de los datasets en train y test y el tratamiento de missings.
* results: contiene las metricas resultantes de modelo, feature importance y pickles de cada uno. Solo se subieron los datos correspondientes a los modelos seleccionados.
* experiments: contiene ejemplos de algunos de los experimentos que se realizaron.

nota: existe una carpeta adicional denominada 'data' donde se almacenan los outputs de los scripts de la carpeta 'etl' que son input del pipeline de entrenamiento; no se comparte en el depositorio para preservar la confidencialidad. 

