from sklearn import metrics, preprocessing
import datetime as dt
import pandas as pd
from imblearn.over_sampling import SMOTE ,RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import numpy as np


def generar_metricas_test(test_y,pred_y,scores):
    tn, fp, fn, tp = metrics.confusion_matrix(test_y,pred_y).ravel()
    accuracy = metrics.accuracy_score(test_y, pred_y)
    balanced_accuracy = metrics.balanced_accuracy_score(test_y, pred_y)
    precision = metrics.precision_score(test_y, pred_y)
    recall = metrics.recall_score(test_y, pred_y)
    f1score = metrics.f1_score(test_y, pred_y)
    roc_auc_score = metrics.roc_auc_score(test_y, pred_y)
    roc_auc_score_train_mean= scores.mean()
    roc_auc_score_train_std= scores.std()
    roc_auc_score_train = str(round(roc_auc_score_train_mean,2)) + "(" + str(round(roc_auc_score_train_std,2)) + ")"
    return(tn, fp, fn, tp,accuracy,balanced_accuracy,precision,recall,f1score,roc_auc_score, roc_auc_score_train)
    


def generar_dataframe_resultados(id_experimento, 
                                 inicio_time,
                                 modelo,
                                 algoritmo,
                                 params,
                                 scaling,
                                 resampling,
                                 tn, fp, fn, tp,
                                 accuracy,
                                 balanced_accuracy,
                                 precision,
                                 recall,
                                 f1score,
                                 roc_auc_score,
                                 roc_auc_score_train,
                                 otras_metricas,
                                 train_name_x,
                                 train_x_shape,
                                 train_name_y,
                                 train_y_shape,
                                 test_name_x,
                                 test_x_shape,
                                 test_name_y,
                                 test_y_shape,
                                 extras):
    
    
    df_resultados_modelo = pd.DataFrame(columns = ['experimento',
                                                   'timestamp',
                                                   'timestamp_dif',
                                                   'modelo',
                                                   'algoritmo',
                                                   'hiperparametros',
                                                   'scaling',
                                                   'resampling',
                                                   'tn', 'fp', 'fn', 'tp',
                                                   'accuracy',
                                                   'balanced_accuracy',
                                                   'precision',
                                                   'recall',
                                                   'f1score',
                                                   'roc_auc_score',
                                                   'roc_auc_score_train',
                                                   'otras_metricas',
                                                   'dataset_trainX',
                                                   'dataset_trainY',
                                                   'dataset_testX',
                                                   'dataset_testY',
                                                   'extras'])
    
        
    df_resultados_modelo = df_resultados_modelo.append({'experimento': id_experimento,
                                                        'timestamp': dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                        'timestamp_dif': dt.datetime.now() - inicio_time,
                                                        'modelo':modelo,
                                                        'algoritmo': algoritmo,
                                                        'hiperparametros':params,
                                                        'scaling':scaling,
                                                        'resampling':resampling,
                                                        'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp,
                                                        'accuracy': accuracy,
                                                        'balanced_accuracy':balanced_accuracy,
                                                        'precision':precision,
                                                        'recall': recall,
                                                        'f1score':f1score,
                                                        'roc_auc_score': roc_auc_score,
                                                        'roc_auc_score_train': roc_auc_score_train,
                                                        'otras_metricas':otras_metricas,
                                                        'dataset_trainX': {'nombre':train_name_x,
                                                                           'shape': train_x_shape},
                                                        'dataset_trainY': {'nombre':train_name_y,
                                                                           'shape': train_y_shape},
                                                        'dataset_testX': {'nombre':test_name_x,
                                                                           'shape': test_x_shape},
                                                        'dataset_testY': {'nombre':test_name_y,
                                                                           'shape': test_y_shape},
                                                        'extras':extras}, ignore_index = True)
    
    return(df_resultados_modelo)


def preprocessing_standarization(scaling, X, test_x):
    
    if scaling == 'Standard':
        # StanderScaler para que converga
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        test_x = scaler.transform(test_x)
    else:
        X = X
        test_x = test_x
    
    return (X,test_x)

def aplicar_resampling(resampling, X, Y):
    sm = SMOTE(random_state=42)
    if resampling == None:
        X, Y = (X, Y)
    elif resampling == 'SMOTE':
        X, Y = sm.fit_resample(X, Y)
    elif resampling == 'random_oversampler':
        ros = RandomOverSampler(random_state=42)
        X, Y = ros.fit_resample(X, Y)
    elif resampling == 'undersamping_knn':
        cc = ClusterCentroids(random_state=42)
        X, Y = cc.fit_resample(X, Y)
    elif resampling == 'undersampling_EditedNearestNeighbours':
        enn = EditedNearestNeighbours(random_state=42)
        X, Y = enn.fit_resample(X, Y)
    elif resampling == 'SMOTEENN':
        smote_enn = SMOTEENN(random_state=42)
        X, Y = smote_enn.fit_resample(X, Y)
    else:
        X, Y = (X, Y)
    return(X, Y)

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

    
    
    