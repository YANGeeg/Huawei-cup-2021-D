#This is a script to do pre-processing on the EEG data
import numpy as np
import math
import h5py
import os
from pathlib import Path
import scipy.io as sio

class Processer:
    def __init__(self):
        self.data = None
        self.label = None
        self.data_processed = None
        self.label_processed = None

    def load_data(self, path):
        path = Path(path)

        # file_codedata = 'd' + '.mat'
        # filedata = path / file_codedata
        # file_codelabel = 'lsvr' + '.mat'

        file_codedata = 'Data_all' + '.mat'
        filedata = path / file_codedata
        file_codelabel = 'L_m5' + '.mat'
        filelabel = path / file_codelabel
        Data = sio.loadmat(filedata)
        Label = sio.loadmat(filelabel)
        data = Data['Data_all']
        label = Label['L_m5']
        # data = Data['d']
        # label = Label['lsvr']

        data_list = []
        label_list = []
        data_list.append(data)
        label_list.append(label)
        print('The shape of data is:' + str(data_list[-1].shape))
        print('The shape of label is:' + str(label_list[-1].shape))
        self.data = np.stack(data_list, axis=0)  # 堆叠，把被试叠加
        self.label = np.stack(label_list, axis=0)
        # data: subject x trial x channels x datapoint
        # label: subject x trial x datapoint
        print('The shape of data is:' + str(data.shape))
        print('The shape of label is:' + str(label.shape))
        print('***************Data loaded successfully!***************')

    def format_data(self):
        # data: subject x trial x channels x datapoint
        # label: subject x trial x datapoint
        data = self.data
        label = self.label
        print('The shape of data is:' + str(data.shape))
        print('The shape of label is:' + str(label.shape))
        # change the label representation 1.0 -> 0.0; 2.0 -> 1.0
        # label[label == 1.0] = 0.0
        # label[label == 2.0] = 1.0

        # Expand the frequency dimention
        # print(data.shape[2])   ##1*432*30*1024

        self.data_processed = np.expand_dims(data, axis=2)  # 在data的第三个维度前加一

        self.label_processed = label

        print("The data shape is:" + str(self.data_processed.shape))

    def split_data(self,  save=True, num=405):
        # data: subject x trial x 1 x channels x datapoint
        # label: subject x trial x datapoint
        # Parameters
        data = self.data_processed
        label = self.label_processed
        # Split the data given
        data_shape = data.shape  # 没用到
        label_shape = label.shape
        data_step = num # 切片移动步长，向下取整
        data_segment = num  # 切片步长
        data_split = []
        label_split = []

        number_segment =1  ###切片个数
        for i in range(number_segment):
            data_split.append(data[:, :, :, :, (i * data_step):(i * data_step + data_segment)])
            label_split.append(label[:, :, (i * data_step)])
        data_split_array = np.stack(data_split, axis=2)
        label_split_array = np.stack(label_split, axis=2)
        print("这才是保存的The data and label are splited: Data shape:" + str(data_split_array.shape) + "Label:" + str(
            label_split_array.shape))

        ###增加没用到
        np.random.seed(0)
        data = np.concatenate(data_split_array, axis=0)

        label = np.concatenate(label_split_array, axis=0)
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        # data : segments x 1 x channel x data
        # label : segments
        index = np.arange(data.shape[0])
        index_randm = index
        np.random.shuffle(index_randm)
        label = label[index_randm]
        data = data[index_randm]
        print("The 没用到 data and label : Data shape:" + str(data.shape) + " Label:" + str(
            label.shape))

        self.data_processed = data_split_array
        self.label_processed = label_split_array
        

        #TODO: Save the processed data here
        if save == True:
            if self.data_processed.all() != None:
              
              save_path = Path(os.getcwd())    #获得当前路径

              # filename_data = save_path / Path('F:/smy/HuaweiDemo/data_forSVR1.hdf')

              filename_data = save_path / Path('F:/smy/HuaweiDemo/data_forL5.hdf')

              save_data = h5py.File(filename_data, 'w') #write

              save_data['data'] = self.data_processed
              save_data['label'] = self.label_processed
              save_data.close()
              print("Data and Label saved successfully! at: " + str(filename_data))
            else :
              print("data_splited is None")
        

        
if __name__ == "__main__":
    Pro = Processer()
    Pro.load_data(path='F:/smy/HuaweiDemo')  ##location of folder
    Pro.format_data()
    Pro.split_data(save=True,num=405)
