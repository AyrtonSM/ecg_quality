import pandas as pd
import numpy as np
import collections
# from numba import cuda, jit
import pickle5 as pickle
import six
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
        
    # @cuda.jit
    # @jit(nopython=True)
    def build_neural_dataset_csv(self, filepath, filename, ecg):
        data = self.get_dataframe()
        interval_1 = data[['begin_1','end_1','class_1']].dropna() 
        interval_2 = data[['begin_2','end_2','class_2']].dropna()
        interval_3 = data[['begin_3','end_3','class_3']].dropna()
        interval_4 = data[['begin_4','end_4','class_4']].dropna()
        
        intervals = [interval_1, interval_2, interval_3, interval_4]
        limit = 2
        
        print('iniciando...')
        for index, interval in enumerate(intervals):
            main_list = []
            classes = []
            longest = 0
            l_index = 0
            for row in range(len(interval)):
                # print('ok...')
                annotation_info = interval.iloc[row]
                
                list_size = int(annotation_info[1]) - int(annotation_info[0])
                selected_interval_in_ecg = [e/1000000 for e in ecg[int(annotation_info[0]):int(annotation_info[1])]]
                if(list_size > longest):
                    longest = list_size
                    l_index = row

                classes.append(annotation_info[2])
                main_list.append(selected_interval_in_ecg)
                selected_interval_in_ecg = None
                if(limit == 0):
                    break
                limit -= 1
                
            
            print('Generating File ' + str(index))
            
            ecg_data_file = open(filepath+'interval_'+str(index)+'/'+filename+'.npy', 'wb')
            ecg_class_data_file = open(filepath+'interval_'+str(index)+'/'+filename+'_classes'+'.npy', 'wb')
        
            print('adjusting...')
            import itertools as it
            longest_interval = main_list[l_index]

            for i in main_list:
                left_to_be_completed = longest - len(i)
                for j in range(left_to_be_completed):
                    i.append(0)
            
            main_list = np.array(main_list)
            np.save(ecg_data_file, main_list,allow_pickle=True, fix_imports=True)
            ecg_data_file.close()
            main_list = None
            np.save(ecg_class_data_file, classes,allow_pickle=True, fix_imports=True)
            ecg_class_data_file.close()
            classes = None
            # np.savetxt(filepath+'interval_'+str(index)+'/'+filename+'.csv', main_list, delimiter=",",fmt='%s')
            # np.savetxt(filepath+'interval_'+str(index)+'/'+filename+'_classes.csv',classes, delimiter=",", fmt='%s')

            break
            
