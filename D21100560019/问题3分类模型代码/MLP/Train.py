import torch
import time
import scipy.io as sio
import numpy as np
import h5py
import datetime
import os
import torch.nn as nn
from pathlib import Path
from HuaweiDemo.EEGDataset import *
from torch.utils.data import DataLoader

from HuaweiDemo.oneconv import *
SAVE = 'Saved_Files/'
if not os.path.exists(SAVE):  # If the SAVE folder doesn't exist, create one
    os.mkdir(SAVE)
class TrainModel():
    def __init__(self):
        self.data = None
        self.label = None
        self.result = None
        self.input_shape = None  # should be (eeg_channel, time data point)
        self.model = 'TSnet'
        self.patient = 10
        self.sampling_rate = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        # Parameters: Training process
        self.random_seed = 42
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.num_class = 3
        self.batch_size = 64

        # Parameters: Model
        self.dropout = 0.2
        self.hiden_node = 128
        self.dropout2 = 0.2
        self.hiden_node2 = 128
        self.T = 9
        self.S = 6
        self.Lambda = 1e-6
        self.Lambda2 = 1e-6
        # self.adj=None
        self.GCN_hiden1=32
        self.GCN_hiden2 = 32
        self.GCN_hiden3 = 32


    def load_data(self, path):

        path = Path(path)
        dataset = h5py.File(path, 'r')
        self.data = np.array(dataset['data'])
        self.label = np.array(dataset['label'])

        # The input_shape should be (channel x data)
        self.input_shape = self.data[0, 0, 0].shape

        print('Data loaded!\n Data shape:[{}], Label shape:[{}]'
              .format(self.data.shape, self.label.shape))




    def set_parameter(self, cv, model, number_class, sampling_rate,patient,
                      random_seed, learning_rate, epoch, batch_size,
                      dropout, hiden_node,num_T, num_S, Lambda,hiden_node2,dropout2,
                      ):
        self.model = model
        self.sampling_rate = sampling_rate
        self.patient = patient

        # Parameters: Training process
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.num_epochs = epoch
        self.num_class = number_class
        self.batch_size = batch_size
        self.Lambda = Lambda


        # Parameters: Model
        self.dropout = dropout
        self.hiden_node = hiden_node
        self.dropout2 = dropout2
        self.hiden_node2 = hiden_node2
        self.T = num_T
        self.S = num_S



        # Save to log file for checking
        if cv == "Leave_one_subject_out":
            file = open("result_subject.txt", 'a')

        elif cv == "K_fold":
            file = open("result_k_fold.txt", 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(self.model) +
                   "\n1)number_class:" + str(self.num_class) + "\n2)random_seed:" + str(self.random_seed) +
                   "\n3)learning_rate:" + str(self.learning_rate) + "\n4)num_epochs:" + str(self.num_epochs) +
                   "\n5)batch_size:" + str(self.batch_size) +
                   "\n6)dropout:" + str(self.dropout) + "\n7)sampling_rate:" + str(self.sampling_rate) +
                   "\n8)hiden_node:" + str(self.hiden_node) + "\n9)input_shape:" + str(self.input_shape) +
                   "\n10)patient:" + str(self.patient) +
                   "\n11)T:" + str(self.T) +"\n12)S:" + str(self.S) + "\n13)Lambda:" + str(self.Lambda) + '\n')
        file.close()

    def K_fold(self):
        save_path = Path(os.getcwd())
        if not os.path.exists(save_path / Path('Result_model/Leave_one_session_out/history')):
            os.makedirs(save_path / Path('Result_model/Leave_one_session_out/history'))
        # Data dimension: subject x trials x segments x 1 x channel x data
        # Label dimension: subject x trials x segments
        data = self.data
        label = self.label

        shape_data = data.shape  # (subject x trials x segments x 1 x channel x data)
        shape_label = label.shape

        subject = shape_data[0]
        trial = shape_data[1]

        channel = shape_data[4]
        print("Train:K_fold \n1)shape of data:" + str(shape_data) + " \n2)shape of label:" +
              str(shape_label) + " \n3)trials:" + str(trial) +" \n5)channel:" + str(channel))

        # Train and evaluate the model subject by subject
        ACC = []
        ACC_mean = []
        for i in range(subject):
            index = np.arange(trial)
            ACC_subject = []
            ACC_session = []

            data_train = data[i, 0:1974, :, :, :, :]
            label_train = label[i, 0:1974, :]

            data_test = data[i, 1974:2024, :, :, :, :]
            label_test = label[i, 1974:2024, :]

            data_train = np.concatenate(data_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)

            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)

            np.random.seed(200)
            np.random.shuffle(data_train)
            np.random.seed(200)
            np.random.shuffle(label_train)

            np.random.seed(200)
            np.random.shuffle(data_test)
            np.random.seed(200)
            np.random.shuffle(label_test)



            # Split the training set into training set and validation set
            data_train, label_train, data_val, label_val = self.split(data_train, label_train)

            # Prepare the data format for training the model  数组转化为Tensor
            data_train = torch.from_numpy(data_train).float()
            label_train = torch.from_numpy(label_train).long()

            data_val = torch.from_numpy(data_val).float()
            label_val = torch.from_numpy(label_val).long()

            # Data dimension: trials x segments x 1 x channel x data--->#data : segments x 1 x channel x data
            data_test = torch.from_numpy(data_test).float()
            label_test = torch.from_numpy(label_test).long()

            # Check the dimension of the training, validation and test set
            print('Training:', data_train.size(), label_train.size())
            print('Validation:', data_val.size(), label_val.size())
            # print('Test:', data_test.size(), label_test.size())

            # Get the accuracy of the model
            ACC_session = self.train(data_train, label_train,
                                     data_test, label_test,
                                     data_val, label_val,
                                     subject=i,
                                     cv_type="K_fold")

            ACC_subject.append(ACC_session)

            ACC_subject = np.array(ACC_subject)
            mAcc = np.mean(ACC_subject)
            std = np.std(ACC_subject)

            print("Subject:" + str(i) + "\nmACC: %.2f" % mAcc)
            print("std: %.2f" % std)

            # Log the results per subject
            file = open("result_session.txt", 'a')
            file.write('Subject:' + str(i) + ' MeanACC:' + str(mAcc) + ' Std:' + str(std) + '\n')
            file.close()

            ACC.append(ACC_subject)    ###ACC容器获得ACC值
            ACC_mean.append(mAcc)

        self.result = ACC
        # Log the final Acc and std of all the subjects
        file = open("result_session.txt", 'a')
        file.write("\n" + str(datetime.datetime.now()) + '\nMeanACC:' + str(np.mean(ACC_mean)) + ' Std:' + str(
            np.std(ACC_mean)) + '\n')
        file.close()

        print("Mean ACC:" + str(np.mean(ACC_mean)) + ' Std:' + str(np.std(ACC_mean)))


        # Save the result
        save_path = Path(os.getcwd())
        filename_data = save_path / Path('Result_model/Result.hdf')
        save_data = h5py.File(filename_data, 'w')
        save_data['result'] = self.result
        save_data.close()

    def split(self, data, label):

        # get validation set
        val = data[int(data.shape[0] * 0.9):]
        val_label = label[int(data.shape[0] * 0.9):]

        train = data[0:int(data.shape[0] * 0.9)]
        train_label = label[0:int(data.shape[0] * 0.9)]

        return train, train_label, val, val_label

    def make_train_step(self, model, loss_fn, optimizer):
        def train_step(x, y):
            model.train()


            #A_hat = (loss_a)*A_hat
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item() / len(pred)
            # L1 regularization
            loss_r = self.regulization(model, self.Lambda)

            # yhat is in one-hot representation;

            loss = loss_fn(yhat, y) + loss_r

            # loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item(), acc
        return train_step

    def regulization(self, model, Lambda):
        w = torch.cat([x.contiguous().view(-1) for x in model.parameters()])
        err = Lambda * torch.sum(torch.abs(w))
        return err


    def train(self, train_data, train_label, test_data, test_label,
              val_data,val_label, subject, cv_type):

        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        # Train and validation loss
        losses = []
        accs = []

        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []

        test_losses = []
        test_acc = []
        Acc_test = []

        # hyper-parameter
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs

        # build the model
        if self.model == 'TSCN2':
            model = TSCN2(num_classes = self.num_class, input_size = self.input_shape,
                            num_T=self.T, num_S=self.S,
                             hiden = self.hiden_node, dropout_rate = self.dropout,
                          hiden2=self.hiden_node2, dropout_rate2=self.dropout2)
        elif self.model == 'TSCN':

            model = TSCN(num_classes = self.num_class, input_size = self.input_shape,
                            num_T = self.T, num_S = self.S,
                            hiden = self.hiden_node, dropout_rate = self.dropout)

        #########改变反向传播算法
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        #########

        loss_fn = nn.NLLLoss()  ###交叉熵
        # loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step(model, loss_fn, optimizer)

        # load the data
        dataset_train = EEGDataset(train_data, train_label)  ##网络中的x,y就是

        dataset_test = EEGDataset(test_data, test_label)

        dataset_val = EEGDataset(val_data, val_label)

        # Dataloader for training process
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=False)

        val_loader = DataLoader(dataset=dataset_val, batch_size=self.batch_size, pin_memory=False)

        test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, pin_memory=False)

        total_step = len(train_loader)

        ####### Training process ########
        Acc = []
        acc_max = 0

        for epoch in range(num_epochs):
            loss_epoch = []
            acc_epoch = []
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)

            losses.append(sum(loss_epoch) / len(loss_epoch))
            accs.append(sum(acc_epoch) / len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, losses[-1], accs[-1]))

            ######## Validation process ########
            predsave2 = []
            trueSVR=[]
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    # adj = torch.from_numpy(self.adj).float()
                    # adj=adj.to(self.device)
                    # A_hat = self.adj.to(self.device)
                    model.eval()

                    yhat = model(x_val)
                    pred = yhat.max(1)[1]  ##索引
                    correct = (pred == y_val).sum()
                    acc = correct.item() / len(pred)

                    # y_val=y_val.float()

                    val_loss = loss_fn(yhat, y_val)
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)
                    predsave2.append(pred)
                    trueSVR.append(y_val)

                Acc_val.append(sum(val_acc) / len(val_acc))
                Loss_val.append(sum(val_losses) / len(val_losses))
                print('Evaluation Loss:{:.4f}, Acc: {:.4f}'
                      .format(Loss_val[-1], Acc_val[-1]))
                val_losses = []
                val_acc = []

                # ndarray2 = np.array([])
                # print(len(predsave2))
                # for i in range(len(predsave2)):
                #     a = predsave2[i].cpu()
                #     a = a.numpy()
                #     print(a)
                #     ndarray2 = np.concatenate((ndarray2, a), axis=0)
                # print(ndarray2.shape)
                #
                # ndarray3 = np.array([])
                # print(len(trueSVR))
                # for i in range(len(trueSVR)):
                #     a = trueSVR[i].cpu()
                #     a = a.numpy()
                #     print(a)
                #     ndarray3 = np.concatenate((ndarray3, a), axis=0)
                # print(ndarray3.shape)

            # ######## early stop ########
            Acc_es = Acc_val[-1]

            if Acc_es > acc_max:
                acc_max = Acc_es
                patient = 0
                print('----Model saved!----')
                torch.save(model,'max_model1.pt')
            else :
                patient += 1
            if patient > self.patient:
                print('----Early stopping----')
                break

        predsave = []
        ytestsave = []
        predall=[]
        ######## test process ########

        model = torch.load('max_model1.pt')

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                model.eval()
                yhat = model(x_test)
                pred = yhat.max(1)[1]
                correct = (pred == y_test).sum()
                acc = correct.item() / len(pred)
                test_loss = loss_fn(yhat, y_test)
                test_losses.append(test_loss.item())
                test_acc.append(acc)
                predsave.append(pred)

            print('Test Loss:{:.4f}, Acc: {:.4f}'
                  .format(sum(test_losses) / len(test_losses), sum(test_acc) / len(test_acc)))
            Acc_test = (sum(test_acc) / len(test_acc))

            ndarray = np.array([])
            print(len(predsave))
            for i in range(len(predsave)):
                a = predsave[i].cpu()
                a = a.numpy()
                print(a)
                ndarray = np.concatenate((ndarray, a), axis=0)
            print(ndarray.shape)

            labels1=ndarray
            sio.savemat('../HuaweiDemo/preL1.mat', {'labels1': labels1})

        # save the loss(acc) for plotting the loss(acc) curve
        save_path = Path(os.getcwd())
        if cv_type == "K_fold":
            filename_callback = save_path / Path('Result_model/Leave_one_session_out/history/'
                                                 + 'history_subject_' + str(subject)
                                                 + '_history.hdf')
            save_history = h5py.File(filename_callback, 'w')
            save_history['acc'] = accs
            save_history['val_acc'] = Acc_val
            save_history['loss'] = losses
            save_history['val_loss'] = Loss_val
            save_history.close()
        return Acc_test

start = time.time()

train = TrainModel()
# train.load_data('/home/sll/smy/myTGCN/data_s1_split.hdf')  ####地址

train.load_data('../HuaweiDemo/data_forL1.hdf')  ####地址

# Please set the parameters here.
train.set_parameter(cv='K_fold',
                    #TSception  GCN  TGCN4 AMCNNDGCN  TGCN_withGCNlayer5
                    model='TSCN',
                    number_class=2,
                    sampling_rate=405,
                    random_seed=42,
                    learning_rate=0.01,
                    epoch=100,
                    batch_size=16,
                    dropout=0.2,
                    hiden_node=128,
                    hiden_node2=128,
                    dropout2=0.2,
                    patient=10,
                    num_T=12,
                    num_S=18,
                    Lambda=0.00001,
                    )
train.K_fold()


end = time.time()
print("循环运行时间:%.2f秒" % (end - start))







