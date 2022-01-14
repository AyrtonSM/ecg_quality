import pandas as pd
import numpy as np
# import math
import tensorflow as tf
from tensorflow import keras
# from tensorflow import math
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import layers, optimizers
from tensorflow.keras.optimizers import Adam, SGD,RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy,CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Conv2D,Conv1D, MaxPooling2D,MaxPooling1D, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.metrics import SparseCategoricalAccuracy,CategoricalAccuracy
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score   
# import numba          
import collections      
import random 
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_PATH = 'C:\\Users\\ayrto\Downloads\Compressed\\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\\data'       
print(tf.config.list_physical_devices('GPU'))
# print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
print(tf.test.is_built_with_cuda())                                                                                                                                                   
class ECGQualityPredictor:
    log = {}
    # Creating the conv layers and pooling layers
    def generate_conv_layer(self,x,filter_size, kernel_size, stride):
        
        model = Sequential()

        model.add(layers.Conv1D(filter_size, kernel_size,strides=stride, padding="same", activation="relu", input_shape=x))        
        model.add(layers.MaxPooling1D(kernel_size))
        model.add(layers.Flatten())
        
        return model

    def build_dense_layer(self, model, mean_val = 0., std_val=1.,neuron=1, dropout=0.2):

        initializer = RandomNormal(mean=mean_val, stddev=std_val)
        
        model.add(Dense(units=neuron,activation="relu",kernel_initializer=initializer,use_bias=True, bias_initializer='zeros', input_dim=1000)) 
        model.add(layers.LeakyReLU()) 
        model.add(Dropout(dropout))
       
        model.add(Dense(units=4,activation="softmax"))
    
        return model
    
    def build_network(self, data_mean, stddev_mean, neuron, dropout, filter_size, kernel_size, stride):

        model = self.generate_conv_layer((1000,1),filter_size, kernel_size,stride)
        model = self.build_dense_layer(model,data_mean, stddev_mean,neuron, dropout)
        # lr=0.0030000006482499
        model.compile(optimizer=Adam(lr=0.0030000006482499 , decay=0.00030000006482499, clipvalue=0.23), 
                        loss=SparseCategoricalCrossentropy(),
                        metrics=[SparseCategoricalAccuracy()])
        print(model.summary())

        return model

    # @numba.jit(nopython=True, parallel=True)
    def findBestModel(self, hyper_params, model_design):

        BATCHES_KEY     = 'batch_size'
        NEURONS_KEY     = 'neurons'
        DROPOUT_KEY     = 'dropout'
        FILTER_SIZE_KEY = 'filter_size'
        KERNEL_SIZE_KEY = 'kernel_size'
        STRIDE_KEY      = 'stride'

        ACCURACY  = 'test_accuracy'
        LOSS  = 'test_loss'
        
        VAL_ACCURACY  = 'val_accuracy'
        VAL_LOSS  = 'val_loss'
        
        PRECISION  = 'precision'
        RECALL  = 'recall'
        F1_SCORE = 'f1_score'
        
        

        batch      = hyper_params['batch_size']
        neuron      = hyper_params['neurons']
        dropout     = hyper_params['dropout']
        filter_size = hyper_params['filter_size']
        kernel_size = hyper_params['kernel_size']
        stride      = hyper_params['stride']
        
        

        model = self.build_network(model_design['data']['mean'], model_design['data']['stddev'], neuron, dropout, filter_size, kernel_size, stride)
            

        earlyStoppingCallback = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        checkpoint = ModelCheckpoint('data/model-ECG.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='min')
        # classifier, model = Excecutor.exec(model,[earlyStoppingCallback,checkpoint],model_design,batch)
        # with tf.device('/GPU:1'):
        classifier = model.fit(
                        model_design['data']['X_train'],
                        model_design['data']['y_train'],
                        validation_data=(model_design['data']['X_val'], model_design['data']['y_val']),
                        batch_size=batch, #melhor 630 para,2664 ,1584
                        epochs=30,
                        shuffle=True,
                        callbacks = [earlyStoppingCallback,checkpoint])
        
        print(classifier.history)
        score_under_testset = model.evaluate(model_design['data']['X_test'], model_design['data']['y_test'], batch_size=batch)
        
        predictions = model.predict(model_design['data']['X_test'],verbose=0)
        real_classes = [c[0][0] for c in model_design['data']['y_test']]
        predicted_classes = [np.argmax(x) for x in predictions]
        confused = confusion_matrix(real_classes,predicted_classes)

        precision = precision_score(real_classes, predicted_classes, average='macro')
        recall = recall_score(real_classes, predicted_classes, average='macro')
        score = f1_score(real_classes, predicted_classes, average='macro')
        
        print(confused)
        print('Precision: %.5f' % precision)
        print('Recall: %.5f' % recall)
        print('F-Measure: %.5f' % score)

        print(f'Model[batch({batch})_neurons({neuron})_filter-size({filter_size})_stride({stride})_dropout({dropout})_kernel-size({kernel_size})]',' ==> ',f'Erro: {score_under_testset[0]}, Accuracy {score_under_testset[1]*100}') 

        if(BATCHES_KEY in self.log):
            self.log[BATCHES_KEY].append(batch)
        else:
            self.log[BATCHES_KEY] = [batch]

        if(NEURONS_KEY in self.log):
            self.log[NEURONS_KEY].append(neuron)
        else:
            self.log[NEURONS_KEY] = [neuron]
        
        if(DROPOUT_KEY in self.log):
            self.log[DROPOUT_KEY].append(dropout)
        else:
            self.log[DROPOUT_KEY] = [dropout]
        
        if(FILTER_SIZE_KEY in self.log):
            self.log[FILTER_SIZE_KEY].append(filter_size)
        else:
            self.log[FILTER_SIZE_KEY] = [filter_size]
        
        if(KERNEL_SIZE_KEY in self.log):
            self.log[KERNEL_SIZE_KEY].append(kernel_size)
        else:
            self.log[KERNEL_SIZE_KEY] = [kernel_size]
        
        if(STRIDE_KEY in self.log):
            self.log[STRIDE_KEY].append(stride)
        else:
            self.log[STRIDE_KEY] = [stride]
        
        if(LOSS in self.log):
            self.log[LOSS].append(score_under_testset[0])
        else:
            self.log[LOSS] = [score_under_testset[0]]

        if(ACCURACY in self.log):
            self.log[ACCURACY].append(score_under_testset[1])
        else:
            self.log[ACCURACY] = [score_under_testset[1]]

        if(VAL_LOSS in self.log):
            self.log[VAL_LOSS].append(np.mean(classifier.history['val_loss'][0]))
        else:
            self.log[VAL_LOSS] = [np.mean(classifier.history['val_loss'][0])]

        if(VAL_ACCURACY in self.log):
            self.log[VAL_ACCURACY].append(np.mean(classifier.history['val_sparse_categorical_accuracy'][0]))
        else:
            self.log[VAL_ACCURACY] = [np.mean(classifier.history['val_sparse_categorical_accuracy'][0])]
        
        if(PRECISION in self.log):
            self.log[PRECISION].append(precision)
        else:
            self.log[PRECISION] = [precision]
        
        if(RECALL in self.log):
            self.log[RECALL].append(recall)
        else:
            self.log[RECALL] = [recall]
        
        if(F1_SCORE in self.log):
            self.log[F1_SCORE].append(score)
        else:
            self.log[F1_SCORE] = [score]
            

                                    
        
        # logDataframe = pd.DataFrame(data=log)
        # print(logDataframe)
        # logDataframe.to_csv('data/results_models_log.csv') 
                                       

                                        
        # model = self.build_network(data_mean, stddev_mean)
        # earlyStoppingCallback = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        # checkpoint = ModelCheckpoint('data/model-ECG.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='min')
        # classifier = model.fit(
        #                 X_train,
        #                 y_train,
        #                 validation_data=(val_sample, val_label),
        #                 batch_size=50, #melhor 630 para,2664 ,1584
        #                 epochs=5,
        #                 shuffle=True,
        #                 callbacks = [earlyStoppingCallback,checkpoint])
        
        # classifier_data = model.to_json()
        # with open(f'data/ecg_classifier.json','w') as classifier_json:
        #     classifier_json.write(classifier_data)

        # results = model.evaluate(test_sample, test_label, batch_size=50)
        # print(results)


    def __evaluate(self, data, classes, k=10, scoring = 'accuracy'):
        file = open('data/ecg_classifier.json','r')
        classifier = model_from_json(file.readline())
        classifier.load_weights('data/model-ECG.h5')
        # classifier = KerasClassifier(
        #                 build_fn = self.build_network,
        #                 epochs=30,
        #                 batch_size = 250
        #                 )

        results = cross_val_score(estimator = classifier,
                                    X = data,
                                    y = classes,
                                    cv = k,
                                    scoring='accuracy')

        return results

    # @numba.jit(nopython=True)
    def build(self, data):

        # data = dataframe.to_numpy()
        data = shuffle(data)
        train_data = data.loc[:,:999].to_numpy() # Pega todas as linhas e todas as colunas exceto a ultima
        class_data = data[1000].to_numpy()
        
        validation = pd.read_csv(f'{DATA_PATH}/validation/validation.csv',header=None).astype('float32')
        val_sample = validation.loc[:,:999].to_numpy()
        val_label = validation[1000].to_numpy()

        test = pd.read_csv(f'{DATA_PATH}/test/test.csv',header=None).astype('float32')
        test_sample = test.loc[:,:999].to_numpy()
        test_label = test[1000].to_numpy()
    
        data_mean = data.mean().sum()/len(data.mean())
        stddev_mean = data.std().sum()/len(data.mean())   
        

        

        X_train = train_data.reshape(train_data.shape[0],train_data.shape[1],1).astype("float32")
        y_train = class_data.reshape(class_data.shape[0],1,1).astype("float32")

        val_sample = val_sample.reshape(val_sample.shape[0],val_sample.shape[1],1).astype("float32")
        val_label = val_label.reshape(val_label.shape[0],1,1).astype("float32")
        
        test_sample = test_sample.reshape(test_sample.shape[0],test_sample.shape[1],1).astype("float32")
        test_label =  test_label.reshape(test_label.shape[0],1,1).astype("float32")
        # X_train = train_data
        # y_train = class_data

        # val_sample = val_sample
        # val_label = val_label
        
        # test_sample = test_sample
        # test_label =  test_label

        
        
        batch_sizes = [50,150,300,650]
        # neurons     = [500,1250,1750]
        neurons     = [250,500,1000, 1750,2500]
        filter_size = [32,64,128]
        kernel_size = [4,6,8]
        stride      = [4,6]
        dropout     = [0.2,0.4]
        # batch_sizes = [50]
        # neurons     = [64]
        # dropout     = [0.4]
        # filter_size = [64]
        # kernel_size = [6]
        # stride      = [6]

        hyper = {
            'batch_size' : batch_sizes,
            'neurons'    : neurons,
            'dropout'    : dropout,
            'filter_size': filter_size,
            'kernel_size': kernel_size,
            'stride'     : stride
            }

        model_design = {
            'data'       : {
                'X_train'  : X_train,
                'y_train'  : y_train,
                'X_test'   : test_sample,
                'y_test'   : test_label,
                'X_val'    : val_sample,
                'y_val'    : val_label,
                'mean'     : data_mean,
                'stddev'   : stddev_mean
            }
        }

        num_trials = 100
        for i in range(num_trials):
            hyper_params = {}
            for h, values in hyper.items():
                hyper_params[h] = values[random.randint(0, len(values) - 1)]
            print(f'#{num_trials}')
            self.findBestModel(hyper_params, model_design)

        logDataframe = pd.DataFrame(data=self.log)
        print(logDataframe)
        logDataframe.to_csv('data/results_models_log.csv') 


    def build_evaluation(self, data):

        data = shuffle(data)
        train_data = data.loc[:,:999].to_numpy() # Pega todas as linhas e todas as colunas exceto a ultima
        class_data = data[1000].to_numpy()

        print('previsores =>',train_data.shape[0],train_data.shape[1])
        print('classes =>',class_data.shape[0])

        X_train = train_data.reshape(train_data.shape[0],train_data.shape[1],1).astype("float32")
        y_train = class_data.reshape(class_data.shape[0],1).astype("float32")
       

        results = self.__evaluate(X_train, y_train, k=5)
    
        if(results.mean() > 0.80):
            print('Ok, conseguimos algo bom!')
        else:
            print('E la vamos nós')
        
        print(results)
        print('media', results.mean())
        print('std', results.std())
    
    def load_neural_network(self):
        file = open('data/ecg_classifier.json','r')
        classifier = model_from_json(file.readline())
        classifier.load_weights('data/model-ECG.h5')

        # CLASSE 2
        # df = pd.read_csv('C:/Users/ayrto/Downloads/Compressed/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/data/interval_1/124001_part_110.csv',header=None)
        
        # CLASS 1
        # df = pd.read_csv('C:/Users/ayrto/Downloads/Compressed/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/data/interval_0/126001_part_15.csv',header=None)
        
        # CLASS 3
        df = pd.read_csv('C:/Users/ayrto/Downloads/Compressed/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/data/test/test.csv',header=None).astype('float32')
        
        sample = df.loc[6350:6351,:999].to_numpy()
        # print(sample)

        sample = sample.reshape(sample.shape[0],sample.shape[1],1)
        result = classifier.predict(sample)
        print(result)
        # print(collections.Counter([np.argmax(r) for r in result]))
        # print([np.argmax(r) for r in result])


