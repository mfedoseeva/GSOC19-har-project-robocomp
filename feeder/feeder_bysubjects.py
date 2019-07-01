# sys
import pickle

# torch
# import torch
# import torch.utils.data

import numpy as np
import os

_SUBJECTS = 4
_C = 3
_T = 2000
_V = 15

class Feeder():
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V)
        label_path: the path to label
        center: If true, center at torso
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 #center=False,
                 mmap=False,
                 max_body = 1
                 ):
        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        self.mmap = mmap

        self.load_data()
        
    def load_data(self):
        # data: N C V T 

        # load label
        self.sample_name = {}
        self.label = {}
        self.data = {}
        self.valid_frame_num = {}
        self.N = {}
        self.total_samples = 0

        for i in range(_SUBJECTS):

            data_path_s = os.path.join(self.data_path, str(i))
            label_path_s = os.path.join(self.label, str(i))
            num_frame_path_s = os.path.join(self.num_frame_path, str(i))
            if '.pkl' in label_path_s:
                try:
                    with open(self.label_path_s) as f:
                        self.sample_name['subject' + str(i)], self.label['subject' + str(i)] = pickle.load(f)
                except:
                    # for pickle file from python2
                    with open(self.label_path, 'rb') as f:
                        self.sample_name['subject' + str(i)], self.label['subject' + str(i)] = pickle.load(
                            f, encoding='latin1')
            else:
                raise ValueError()



            # load data
            if self.mmap == True:
                self.data['subject' + str(i)] = np.load(self.data_path,mmap_mode='r')
            else:
                self.data['subject' + str(i)] = np.load(self.data_path,mmap_mode=None) 

            # load num of valid frame length
            self.valid_frame_num['subject' + str(i)] = np.load(self.num_frame_path)

            self.N['subject' + str(i)] = self.data['seubject' + str(i)].shape[0]
            self.total_samples += self.N['subject' + str(i)]

        if self.max_body != 1 :
            raise NotImplementedError('multiperson not implemented')

        self.all_label = np.array(list(self.label.values()))
        self.all_valid_frames_num = np.array(list(self.valid_frame_num.values()))
        self.all_sample_name = np.array(list(self.sample_name.values()))



    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    # def __getitem__(self, index):
    #     # get data
    #     # input: C, T, V
    #     data_numpy = self.data[index]
       
    #     if self.mmap:
    #         data_numpy = np.array(data_numpy) # convert numpy.core.memmap.memmap to numpy


    #     label = self.label[index]
    #     # valid_frame_num = self.valid_frame_num[index]

    #     # if self.center == True:
    #     #     for t in range(valid_frame_num):
    #     #         # coords of the torso
    #     #         torso_coord = data_numpy[:, t, 2]
    #     #         for v in range(self.V):
    #     #             data_numpy[:, t, v] -= torso_coord

    #     return data_numpy, label

    def get_train_and_test(test_subject):
        '''
        for each fold of cross-validation, returns train and test data, labels and valid_frame_number
        '''
        train_subs = [x for x in range(4) if x != test_subject]
        test_data = self.data['subject' + str(test_subject)]
        test_labels = self.label['subject' + str(test_subject)]
        test_valid_frame_num = self.valid_frame_num['subject' + str(test_subject)]
        train_samples = self.total_samples - self.N['subject' + str(test_subject)]
        train_data = np.zeros((train_samples, _C, _T, _V))
        train_labels = np.empty(train_samples)
        train_valid_frame_num = np.zeros(train_samples)
        counter = 0
        print(train_subs)
        for i in train_subs:
            num_samples = self.N['subject' + str(i)]
            train_data[counter : (counter + num_samples), :, :, :] = self.data['subject' + str(i)]
            train_labels[counter : (counter+ num_samples), :, :, :] = self.label['subject' + str(i)]
            train_valid_frame_num[counter : (counter + num_samples), :, :, :] = self.valid_frame_num['subject' + str(i)]
            counter += num_samples
        return train_data, train_labels, train_valid_frame_num, test_data, test_labels, test_valid_frame_num

if __name__ == '__main__':

    import numpy as np

    # testing
    base_path = "../data0/CAD-60"
    environments = ['bathroom', 'bedroom', 'kitchen', 'livingroom', 'office']
    data_file = "train_data.npy"
    label_file = "train_label.pkl"
    num_frame_file = "train_num_frame.npy"
    
    for env in environments:

        print(f"Environment: {env}")

        data_path = os.path.join(base_path, env, data_file)
        label_path = os.path.join(base_path, env, label_file)
        num_frame_path = os.path.join(base_path, env, num_frame_file)

        dataset = Feeder(data_path, label_path, num_frame_path)

        print('Labels distribution: ')
        oneh_vector = np.zeros(12)
        bins = np.bincount(dataset.all_label)
        for i in range(len(bins)):
            oneh_vector[i] = bins[i] 
        print(oneh_vector)

        # only print for small datasets
        for i, j in zip(dataset.all_sample_name, dataset.all_label):
            print(str(i) + ' | ' + str(j))

        print('Num of frames in samples: ')
        print(dataset.all_valid_frame_num)

        print('Samples statistics: ')
        print(f"{dataset.N} samples")
        print(f"{np.mean(dataset.all_valid_frame_num):.2f} average frames") 
        print ('------------------')












