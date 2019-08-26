from feeder.feeder import Dataset
import sys
sys.path.insert(0, '../')
from component_code.inference import predict_sample
from feeder import utils
import numpy as np
import os
from shutil import copy2

_CLASS_NAMES = ['talking on the phone', 'writing on whiteboard', 'drinking water', 'rinsing mouth with water', 'brushing teeth', 'wearing contact lenses',
 'talking on couch', 'relaxing on couch', 'cooking (chopping)', 'cooking (stirring)', 'opening pill container', 'working on computer']
data_path = '../../cad60dataset'

if __name__ == '__main__':
    params = utils.Params()
    params.data_feeder_args["data_path"] = './data0/CAD-60/all/train_data.npy'
    params.data_feeder_args["num_frame_path"] = './data0/CAD-60/all/train_num_frame.npy'
    params.data_feeder_args["label_path"] = './data0/CAD-60/all/train_label.pkl'
    dataset = Dataset(**params.data_feeder_args)

    num = 5
    X = dataset.data
    print(X.shape)
    Y = dataset.label
    x = X[num, :, :, :]
    y = Y[num]
    idx = [1, 30, 60, 90]
    x = x[:, idx, :]
    np.save('test_sample', x)
    # x = np.load('test_sample.npy')
    names = dataset.sample_name
    print(names[num])
    print(data_path)
    sample_path = data_path + names[num].split('.')[0]
    print(sample_path)
    copy_dir = '../../test_sample_pics'
    if not os.path.exists(copy_dir):
        os.makedirs(copy_dir)
    for i in idx:
        img_path = sample_path + '/' + 'RGB_' + str(i + 1) + '.png'
        copy2(img_path, copy_dir)
    print('true label: ')
    print(_CLASS_NAMES[y])
    # x = np.load('inferred_sample.npy')
    print('prediction top 5: ')
    predict_sample(x)