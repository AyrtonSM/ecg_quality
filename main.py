import pandas as pd
import numpy as np
import collections
from service.DataCleanse import DataCleanse
from service.ECGReading import ECGReading
from service.ai.core.ECGQualityPredictor import ECGQualityPredictor
import math
import matplotlib.pyplot as plt
# import keras
import random
from matplotlib import pyplot as plt
# from numba import cuda, jit
import os
os.environ['AUTOGRAPH_VERBOSITY'] = '10'

def main():


    # # ecg_folders = os.listdir('../quality_ecg_db')[:7]
    ecg_folders = os.listdir('../quality_ecg_db')[:18]


    for sample in ecg_folders:
        path_csv = f"../quality_ecg_db/{sample}/{sample}_ANN.csv"
        path_ecg = f"../quality_ecg_db/{sample}/{sample}_ECG.dat"

        data_cleanse = DataCleanse()
        data_cleanse.cleanse_file(path_csv)

        ecg_reading = ECGReading()
        ecg = ecg_reading.read_ecg(path_ecg)
        # print(ecg[:5])
        data_cleanse.build_neural_dataset_csv_4('../data/', sample , ecg)
    
    data_cleanse = DataCleanse()
    # data_cleanse.merge_all_intervals()
    # data_cleanse.merge_all_majors()
    data_cleanse.build_test_train_validation_data()

        

def build_ai_model():
    print("Loading..")
    ecg_quality_predictor = ECGQualityPredictor()
    # dataframe = pd.read_csv('../data/ecg_all_signals.csv',header=None)
    # dataframe = pd.read_csv('../data/interval_0/100001_part_0.csv',header=None)
    # dataframe = pd.read_csv('../data/interval_3/major_table_interval_3.csv',header=None).astype('float32')
    # dataframe = pd.read_csv('../data/ecg_all_signals.csv',header=None).astype('float32')
    # dataframe = pd.read_csv('../data/validation/validation.csv',header=None).astype('float16')
    dataframe = pd.read_csv('../data/train/train.csv',header=None).astype('float32')
    # dataframe = pd.read_csv('../data/test_subject.csv',header=None)
    print(dataframe)

    # validation = pd.read_csv(f'../data/validation/validation.csv',header=None).astype('float32')
    # val_sample = validation.loc[:,:999].to_numpy()
    # val_label = validation[1000].to_numpy()

    # validation_set = np.array(list(zip(val_sample,val_label)))
    # validation_set = [a for a in zip(val_sample,val_label)]
    # print(validation_set[:5])
    ecg_quality_predictor.build(dataframe)

def predict():
    ecg_quality_predictor = ECGQualityPredictor()
    ecg_quality_predictor.load_neural_network()

def evaluate():
    ecg_quality_predictor = ECGQualityPredictor()
    dataframe = pd.read_csv('../data/validation/validation.csv',header=None).astype('float32')
    ecg_quality_predictor.build_evaluation(dataframe)

def load_report():
    df = pd.read_csv('data/results_models_log.csv').astype('float32')
    df = df.loc[:,'test_loss':].describe(include='all')
    print(df)
    df.to_csv('data/metrics.csv')

def find_best_combination():
    df = pd.read_csv('data/results_models_log.csv')
    print(df)
    val_loss = df['val_loss'].tolist() 
    test_loss = df['test_loss'].tolist() 
    # rs = [x for x in zip(val_loss, test_loss)]
    best_dataframe = df[df.f1_score == df.f1_score.max()]
    best_dataframe.to_csv('data/best_model_by_f1_score.csv')
    worst_dataframe = df[df.f1_score == df.f1_score.min()]
    worst_dataframe.to_csv('data/worst_model_by_f1_score.csv')
    print('best combination => \n' , best_dataframe)
    print('worst combination => \n' , worst_dataframe)
    print(df.f1_score)
    print(f"F1_SCORE_AVG: {np.mean(df.f1_score)}")
    print(f"ACCURACY_AVG: {np.mean(df.val_accuracy)}")
    print(f"TEST_AVG: {np.mean(df.test_accuracy)}")
    print(f"PRECISION_AVG: {np.mean(df.precision)}")
    print(f"RECALL_AVG: {np.mean(df.recall)}")

    # plt.plot([x for x in range(100)],df.f1_score,'go-' ,label='Métrica f1_score por modelo')
    # plt.legend()
    # plt.plot([x for x in range(100)],df.precision,'b--' ,label='Métrica precision por modelo',alpha=0.4)
    # plt.legend()
    # plt.plot([x for x in range(100)],df.recall,'yx-' ,label='Métrica recall por modelo', alpha=0.4)
    # plt.legend()

    # plt.plot([x for x in range(100)],df.test_loss,'go-' ,label='Erro na base de teste por modelo')
    # plt.legend()
    # plt.plot([x for x in range(100)],df.val_loss,'b--' ,label='Erro na base de validação por modelo',alpha=0.4)
    # plt.legend()
    
    # plt.plot([x for x in range(100)],df.test_accuracy,'go-' ,label='Acurácia na base de teste por modelo')
    # plt.legend()
    # plt.plot([x for x in range(100)],df.val_accuracy,'b--' ,label='Acurácia na base de validação por modelo',alpha=0.4)
    # plt.legend()


    # plt.plot([x for x in range(100)],df.recall,'yx-' ,label='Métrica recall por modelo', alpha=0.4)
    # plt.legend()
   
    
    plt.show()

# main()
build_ai_model()
# predict()
# evaluate()
# load_report()
# find_best_combination()