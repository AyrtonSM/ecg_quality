import pandas as pd
import numpy as np
import collections
# from numba import cuda, jit
import pickle5 as pickle
import six
import itertools as it
import glob
import os
from sklearn import preprocessing
from numba import jit   
DATA_PATH = 'C:\\Users\\ayrto\Downloads\Compressed\\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\\data'
class DataCleanse:
    
    def cleanse_file(self,file):
        self.get_dataframe_from_csv_file(file)
    
    def get_self(self):
        return self

    def __columns_adjustment(self, data):
        data = data.shift(periods=1)
        data.loc[0] = [int(float(c)) for c in data.columns]
        data.columns = ['begin_1','end_1','class_1','begin_2','end_2','class_2','begin_3','end_3','class_3','begin_4','end_4','class_4']
        return data
    
    def get_dataframe_from_csv_file(self, file):
        csv = pd.read_csv(file)
        self.__dataframe = self.__columns_adjustment(csv)
        # self.__dataframe = self.__dataframe.fillna(-1)
        return self.__dataframe

    
    def remove_nan_values(self):
        #TO BE IMPLEMENTED
        return None

    def get_dataframe(self):
        return self.__dataframe

    def sort_annotations(self):
        intervals = []
        data = self.get_dataframe()
        for pos in range(data.shape[0]):
            v = np.array(data.iloc[pos])
            interval_classification = []

            interval_limit = int(len(v)/4)

            for i in range(interval_limit):
                interval_1 = interval_limit * i
                interval_2 = interval_limit * i + 1
                classification = interval_limit * i + 2
                
                interval_min_1 = v[interval_1]
                interval_min_2 = v[interval_2]
                
                interval_classification.append((interval_min_1,interval_min_2,v[classification]))
                intervals.append(interval_classification)
                
        ordered = []
        for i in range(len(intervals)):
            p = pd.DataFrame(data=intervals[i])
            p = p.sort_values(by=[0])
            p = p.sort_values(by=[1])
            ordered.append(p.to_numpy())

        return np.array(ordered)
            
    def build_neural_dataset_csv_3(self, filepath, filename, ecg):
            data = self.get_dataframe()
            interval_1 = data[['begin_1','end_1','class_1']].dropna() 
            interval_2 = data[['begin_2','end_2','class_2']].dropna()
            interval_3 = data[['begin_3','end_3','class_3']].dropna()
            interval_4 = data[['begin_4','end_4','class_4']].dropna()
            del data
            intervals = [interval_1, interval_2, interval_3, interval_4]
            # intervals = [interval_2 , interval_4]
     
            print('iniciando...')
            for index, interval in enumerate(intervals):
                
                pagination = 0
                for row in range(len(interval)):
                    INIT = 0
                    LIMIT = 5000
                    INTERVAL = 5000
                    main_list = []
                    annotation_info = interval.iloc[row]
                    interval_length = int(annotation_info[1]) - int(annotation_info[0])
                    # print('esse eh o intervalo', interval_length, LIMIT)
                    if annotation_info[2] > 0:
                        while True:
                            if INIT + INTERVAL < interval_length:
                                # print(INIT, LIMIT)
                                # selected_interval_in_ecg = [e/100000 for e in ecg[INIT:LIMIT]]
                                # selected_interval_in_ecg = ecg[INIT:LIMIT]
                                if(ecg[INIT:LIMIT] != []):
                                    selected_interval_in_ecg = preprocessing.normalize([ecg[INIT:LIMIT]], norm="l2", axis=1).tolist()[0]
                                    selected_interval_in_ecg.append(int(annotation_info[2]))
                                    main_list.append(selected_interval_in_ecg)
                                # print('quando eh menor que...')
                                # print('sizelist => ', len(selected_interval_in_ecg))
                                INIT = LIMIT
                                LIMIT += INTERVAL
                            elif INIT + INTERVAL > interval_length:
                                diff = interval_length - INIT
                                # print(INIT, diff, interval_length, INIT - (5000 - diff))
                                # INIT = INIT - (INTERVAL - diff)
                                # selected_interval_in_ecg = [e/100000 for e in ecg[INIT:interval_length]]
                                # selected_interval_in_ecg = ecg[INIT:interval_length]
                                if(ecg[INIT:interval_length] != []):
                                    selected_interval_in_ecg = preprocessing.normalize([ecg[INIT:interval_length]], norm="l2", axis=1).tolist()[0]
                                    for i in range((INTERVAL - (INTERVAL - diff)), INTERVAL):
                                        selected_interval_in_ecg.append(0.0)
                                    selected_interval_in_ecg.append(int(annotation_info[2]))
                                    print('quando eh maior que...')
                                    print('sizelist => ', len(selected_interval_in_ecg))
                                    main_list.append(selected_interval_in_ecg)
                                break
                            elif INIT + INTERVAL == interval_length:
                                # selected_interval_in_ecg = [e/100000 for e in ecg[INIT:LIMIT]]
                                # selected_interval_in_ecg = ecg[INIT:LIMIT]
                                if(ecg[INIT:LIMIT] != []):
                                    selected_interval_in_ecg = preprocessing.normalize([ecg[INIT:LIMIT]], norm="l2", axis=1).tolist()[0]
                                    selected_interval_in_ecg.append(int(annotation_info[2]))
                                    # print('quando eh igual...')
                                    # print('sizelist => ', len(selected_interval_in_ecg))
                                    main_list.append(selected_interval_in_ecg)
                                break
                            
                        main_list = np.array(main_list, dtype=np.float16)
                        # print(main_list[0])
                        document = filepath+'interval_'+str(index)+'/'+filename+'_part_'+str(pagination)+'.csv'
                        print('Saving Document : ', filename+'_part_'+str(pagination)+'.csv')
                        np.savetxt(document, main_list, delimiter=",",fmt='%s')
                        print('*Document : ', filename+'_part_'+str(pagination)+'.csv', 'saved')
                        del main_list
                        pagination += 1

    # @jit(nopython=True)
    def build_neural_dataset_csv_4(self, filepath, filename, ecg):
            data = self.get_dataframe()
            interval_1 = data[['begin_1','end_1','class_1']].dropna() 
            interval_2 = data[['begin_2','end_2','class_2']].dropna()
            interval_3 = data[['begin_3','end_3','class_3']].dropna()
            interval_4 = data[['begin_4','end_4','class_4']].dropna()
            del data
            intervals = [interval_1, interval_2, interval_3, interval_4]
            # intervals = [interval_3 , interval_4]
     
            print('iniciando...')
            for index, interval in enumerate(intervals):
                
                pagination = 0
                for row in range(len(interval)):
                    # print(row)
                    annotation_info = interval.iloc[row]
                    interval_length = int(annotation_info[1]) - int(annotation_info[0])
                    END = int(annotation_info[1])-1
                    INIT = int(annotation_info[0])-1
                    LIMIT = 1000
                    INTERVAL = 1000
                    main_list = []
                    # print('esse eh o intervalo', 'inicio=>',INIT, END)
                    if annotation_info[2] > 0:
                        while True:
                            # print(row)
                            # print(INIT, LIMIT, intere)
                            if INIT + LIMIT < END:
                                # print(INIT, END)
                                # selected_interval_in_ecg = [e/100000 for e in ecg[INIT:LIMIT]]
                                # selected_interval_in_ecg = ecg[INIT:LIMIT]
                                if(ecg[INIT:(INIT + LIMIT)] != []):

                                    selected_interval_in_ecg = preprocessing.normalize([ecg[INIT:(INIT + LIMIT)]], norm="l2", axis=1).tolist()[0]
                                    selected_interval_in_ecg.append(int(annotation_info[2]))
                                    main_list.append(selected_interval_in_ecg)
                                    # print('<',len(main_list))
                                    # print('esse eh o intervalo', 'inicio=>',INIT,(INIT + LIMIT))
                                # print('quando eh menor que...')
                                # print('sizelist => ', len(selected_interval_in_ecg))
                                # print('before',INIT)
                                INIT = (INIT + LIMIT)
                                # print('after',INIT)

                                # LIMIT += INTERVAL
                            elif INIT + LIMIT > END:
                                diff = (INIT+LIMIT) - END
                                # print('diff => ', diff)
                                # print(INIT,INIT+LIMIT , interval_length, diff)
                                # print(INIT, diff, interval_length, INIT - (5000 - diff))
                                # INIT = INIT - (INTERVAL - diff)
                                # selected_interval_in_ecg = [e/100000 for e in ecg[INIT:interval_length]]
                                # selected_interval_in_ecg = ecg[INIT:interval_length]
                                if(ecg[INIT:END] != []):
                                    # print(INIT, END)
                                    # print('esse eh o intervalo final?', 'inicio=>',INIT,END)
                                    selected_interval_in_ecg = preprocessing.normalize([ecg[INIT:END]], norm="l2", axis=1).tolist()[0]
                                    # print(len(selected_interval_in_ecg),(LIMIT - (LIMIT - diff)), LIMIT)
                                    # for i in range(len(selected_interval_in_ecg), LIMIT):
                                    for i in range(diff):
                                        selected_interval_in_ecg.append(0.0)
                                    selected_interval_in_ecg.append(int(annotation_info[2]))
                                    # print('quando eh maior que...')
                                    # print('sizelist => ', len(selected_interval_in_ecg))
                                    main_list.append(selected_interval_in_ecg)
                                    # print('>',len(main_list))
                                break
                            elif INIT + LIMIT == END:
                                # selected_interval_in_ecg = [e/100000 for e in ecg[INIT:LIMIT]]
                                # selected_interval_in_ecg = ecg[INIT:LIMIT]
                                if(ecg[INIT:(INIT + LIMIT)] != []):
                                    # print(INIT, END)
                                    # print('esse eh o intervalo final quando é igual?', 'inicio=>',INIT,(INIT + LIMIT))
                                    selected_interval_in_ecg = preprocessing.normalize([ecg[INIT:(INIT + LIMIT)]], norm="l2", axis=1).tolist()[0]
                                    selected_interval_in_ecg.append(int(annotation_info[2]))
                                    # print('quando eh igual...')
                                    # print('sizelist => ', len(selected_interval_in_ecg))
                                    main_list.append(selected_interval_in_ecg)
                                    # print('==',len(main_list))
                                break
                        
                        
                        main_list = np.array(main_list, dtype=np.float32)  
                        # print(main_list[0])
                        document = filepath+'interval_'+str(index)+'/'+filename+'_part_'+str(pagination)+'.csv'
                        print('Saving Document : ', filename+'_part_'+str(pagination)+'.csv')
                        np.savetxt(document, main_list, delimiter=",",fmt='%s')
                        print('*Document : ', filename+'_part_'+str(pagination)+'.csv', 'saved')
                        del main_list
                        pagination += 1
                    # print('iniciando mais uma linha...')
    
    def merge_all_intervals(self):
        sample_names = ['100001','100002','103001','103002', '103003',  '104001','105001','111001', '113001', '114001', '115001', '118001', '121001', '122001', '123001', '124001', '125001', '126001']
        # sample_names = ['103001','103003', '104001','105001','122001','124001']
        # sample_names = ['100001']
        # intervals = ['interval_0','interval_1','interval_2','interval_3']
        intervals = ['interval_0','interval_1','interval_2','interval_3']
        # intervals = ['interval_0','interval_1']
        # intervals = ['interval_2']
        extension = 'csv'
        # for interval in intervals:
        print(f"building up {intervals[0]}")
        limit = 0
                
        for i in range(len(intervals)):
            main_array = []
            document = f"{DATA_PATH}\\{intervals[i]}\\major_table_interval_{i}.csv"
            # files = [f for f in glob.glob(f'{DATA_PATH}\\{intervals[i]}\\*.{extension}')]
            for sample in sample_names:
                files = [f for f in glob.glob(f'{DATA_PATH}\\{intervals[i]}\\{sample}_part_*.{extension}')]

                for idx in range(len(files)):
                    filename = f"{DATA_PATH}\\{intervals[i]}\\{sample}_part_{idx}.csv"    
                    print('reading : ', filename)

                    # Check if file is empty or not
                    if(os.stat(filename).st_size != 0):
                        ecg_interval_with_class = list(pd.read_csv(filename).to_numpy())
                        if(len(ecg_interval_with_class) > 0):
                            print(len(ecg_interval_with_class))
                            # main_array.append(ecg_interval_with_class[0])
                            print('saving ==>')
                            for index,row in enumerate(ecg_interval_with_class):
                                print(f'line ==>{index}')
                                
                                with open(document,"a") as ecg_file:
                                    content = ','.join(map(str, row))+'\n'
                                    ecg_file.write(content)

            

            del main_array
                


        print("Done!")
    
    def merge_all_majors(self):
        # intervals = ['interval_0','interval_1','interval_2','interval_3']
        intervals = ['interval_1','interval_3']
        s = [1,3]
        # intervals = ['interval_3']
        extension = 'csv'
        # for interval in intervals:
        print(f"building up {intervals[0]}")
        limit = 0
        main_array = []    
        for i in range(len(intervals)):
         
            filename = f"{DATA_PATH}\\{intervals[i]}\\major_table_interval_{s[i]}.csv"    
            print('reading : ', filename)
            ecg_interval_with_class = list(pd.read_csv(filename).to_numpy())
            if(len(ecg_interval_with_class) > 0):
                # print(len(ecg_interval_with_class))
                print('saving ==>')
                for idx,row in enumerate(ecg_interval_with_class):
                    print(f'line ==>{idx}')
                    with open(f"{DATA_PATH}\\ecg_all_signals.csv","a") as ecg_file:
                        content = ','.join(map(str, row))+'\n'
                        ecg_file.write(content)
                    

        del main_array
                


        print("Done!")
    
    def build_test_train_validation_data(self):
        sample_names = ['100001','100002','103001','103002', '103003',  '104001','105001','111001', '113001', '114001', '115001', '118001', '121001', '122001', '123001', '124001', '125001', '126001']
        # sample_names = ['105001','111001', '113001', '114001', '115001', '118001', '121001', '122001', '123001', '124001', '125001', '126001']
        # intervals = ['interval_0','interval_1','interval_2','interval_3']
        intervals = ['interval_2']
        extension = 'csv'
        # print(f"building up {intervals[0]}")
        limit = 0
                
        for i in range(len(intervals)):
            main_array = []
            # document = f"{DATA_PATH}\\{intervals[i]}\\major_table_interval_{i}.csv"
            # files = [f for f in glob.glob(f'{DATA_PATH}\\{intervals[i]}\\*.{extension}')]
            for sample in sample_names:
                files = [f for f in glob.glob(f'{DATA_PATH}\\{intervals[i]}\\{sample}_part_*.{extension}')]

                

                # document = f"{DATA_PATH}\\{intervals[i]}\\major_table_interval_{i}.csv")


                for idx in range(len(files)):
                    filename = f"{DATA_PATH}\\{intervals[i]}\\{sample}_part_{idx}.csv"
                    
                        

                    if(os.stat(filename).st_size != 0):
                        ecg_interval_with_class = list(pd.read_csv(filename).to_numpy())
                        test_size = int((len(ecg_interval_with_class)-1) * 0.1)
                        val_size = int((len(ecg_interval_with_class)-1) * 0.1)
                        train_size = len(ecg_interval_with_class) - val_size - test_size - 1

                        # print(document)
                        if(len(ecg_interval_with_class) > 0):
                            print(f'saving ==> {filename} from {intervals[i]}')
                            for idx,row in enumerate(ecg_interval_with_class):
                                
                                if(idx < test_size):
                                    document = f"{DATA_PATH}\\test\\test.csv"
                                elif(idx >= test_size and idx < test_size + val_size):
                                    document = f"{DATA_PATH}\\validation\\validation.csv"
                                else:
                                    document = f"{DATA_PATH}\\train\\train.csv"
                                
                                print(document)
                                with open(document,"a") as ecg_file:
                                    content = ','.join(map(str, row))+'\n'
                                    ecg_file.write(content)
