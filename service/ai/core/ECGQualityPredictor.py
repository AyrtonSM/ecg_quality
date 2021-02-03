import pandas as pd
import numpy as np
import tensorflow
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D


class ECGQualityPredictor:

    # Creating the conv layers and pooling layers
    def generate_conv_layer(self,x):
        x = Conv2D(filters=1, kernel_size=(1,3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)

        x = Conv2D(filters=1, kernel_size=(1,3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)

        x = Conv2D(filters=1, kernel_size=(1,3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)

        # Apply global average pooling to get flat feature vectors
        x = layers.GlobalAveragePooling2D()(x)
        return x

    def build_dense_layer(self,conv_layer):
        num_classes = 3
        outputs = layers.Dense(num_classes, activation="softmax")(conv_layer)
        return outputs

    def build_model(self,inputs, outputs):
        model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
        return model


