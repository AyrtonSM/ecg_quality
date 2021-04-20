import pandas as pd
import numpy as np
import collections
from service.DataCleanse import DataCleanse
from service.ECGReading import ECGReading
from service.ai.core.ECGQualityPredictor import ECGQualityPredictor
import math
import matplotlib.pyplot as plt
import keras
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
    val_loss = df['val_loss'].tolist() 
    test_loss = df['test_loss'].tolist() 
    # rs = [x for x in zip(val_loss, test_loss)]
    plt.plot(val_loss, 'go-', label='Loss in Validation dataset', linewidth=2)
    plt.legend()
    plt.plot(test_loss, 'bo-', label='Loss in Test dataset')
    plt.legend()
    
    best_dataframe = df[df.val_loss == df.val_loss.min()]
    best_dataframe.to_csv('data/best_model_by_f1_score.csv')
    worst_dataframe = df[df.val_loss == df.val_loss.max()]
    worst_dataframe.to_csv('data/worst_model_by_f1_score.csv')
    print('best combination => \n' , best_dataframe)
    print('worst combination => \n' , worst_dataframe)

    plt.show()

# main()
# build_ai_model()
# predict()
# evaluate()
# load_report()
find_best_combination()