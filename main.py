import pandas as pd
import numpy as np
import collections
from service.DataCleanse import DataCleanse
from service.ECGReading import ECGReading
from service.ai.core.ECGQualityPredictor import ECGQualityPredictor
import math
import matplotlib.pyplot as plt
import tensorflow as tf
# from numba import cuda, jit
import os


def main():


    ecg_folders = os.listdir('../quality_ecg_db')[:18]

    for sample in ecg_folders:
        path_csv = f"../quality_ecg_db/{sample}/{sample}_ANN.csv"
        path_ecg = f"../quality_ecg_db/{sample}/{sample}_ECG.dat"

        data_cleanse = DataCleanse()
        data_cleanse.cleanse_file(path_csv)

        ecg_reading = ECGReading()
        ecg = ecg_reading.read_ecg(path_ecg)

        data_cleanse.build_neural_dataset_csv('../data/', sample , ecg)
        

def build_ai_model():
    ecg_quality_predictor = ECGQualityPredictor()
    # dataframe = pd.read_csv('../data/interval_0/100001.csv',header=None)
    ecg_data_file = open('../data/interval_0/100002.npy', 'rb')
    ecg_class_data_file = open('../data/interval_0/100002_classes.npy', 'rb')

    ecg_data = np.load(ecg_data_file,allow_pickle=True)
    ecg_class_data = np.load(ecg_class_data_file,allow_pickle=True)

    data_shape = (ecg_data.shape[0],ecg_data.shape[1], 2)
    inputs = tf.keras.Input(shape=data_shape)
    conv_layer = ecg_quality_predictor.generate_conv_layer(inputs)
    outputs = ecg_quality_predictor.build_dense_layer(conv_layer)
    ECGQualityModel = ecg_quality_predictor.build_model(inputs, outputs)

    print(ECGQualityModel.summary())

# main()
build_ai_model()

