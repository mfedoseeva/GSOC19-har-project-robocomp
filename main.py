import argparse
import os
import random
import numpy as np
from feature_extraction.feature_extractor1 import extract_features_type1
from feature_extraction.tools import *
from feeder.feeder import Feeder
from feeder import utils

# for SVM
from sklearn import svm
from sklearn.model_selection import cross_val_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./data0/', help="directory of the dataset")
parser.add_argument('--environment', default='office', help='office, livingroom, kitchen, bedroom or bathroom')

# HERE it is decided which dataset will be used
parser.add_argument('--dataset_name', default='CAD-60', help="dataset name ")
# parser.add_argument('--model_dir', default='./',
#                     help="parents directory of model")

# parser.add_argument('--model_name', default='SVM',help="model name")
# # parser.add_argument('--load_model',
# #         help='Optional, load trained models')
# # parser.add_argument('--load',
# #         type=str2bool,
# #         default=False,
# #         help='load a trained model or not ')
# parser.add_argument('--mode', default='train', help='train or test')
# parser.add_argument('--num', default='01', help='num of trials (type: list)')


def fetch_data(params):
    # creates and returns feeder object with data
    if 'CAD-60' in params.dataset_name:
        params.data_feeder_args["data_path"] = params.dataset_dir+'/CAD-60'+ '/' + params.environment + '/train_data.npy'
        params.data_feeder_args["num_frame_path"] = params.dataset_dir+'/CAD-60' + '/' + params.environment +'/train_num_frame.npy'
        params.data_feeder_args["label_path"] = params.dataset_dir + '/CAD-60' + '/' + params.environment + '/train_label.pkl'
        dataset = Feeder(**params.data_feeder_args)
        return dataset
    else:
        raise NotImplementedError('only CAD-60 is supported')

def custom_cv_subj_fbf(valid_frame_num, subjects=4):
    # custom cross-validation rule, each fold = 1 subject
    n = len(valid_frame_num)
    actions = n/subjects
    
    frames_per_subject = np.zeros(subjects, dtype=int)
    for i in range(n):
        # subj from 0 to 3
        subj = int(i/actions)
        frames_per_subject[subj] += valid_frame_num[i]
    
    subj_end_idx = [None]*subjects

    prev = 0
    for i in range(subjects):
        subj_end_idx[i] = frames_per_subject[i] + prev
        prev += frames_per_subject[i] 

    start = 0
    for i in range(subjects):
        val_idx = np.arange(start, subj_end_idx[i])
        start += frames_per_subject[i]
        train_idx = [x for x in range(sum(valid_frame_num)) if x not in val_idx]
        yield train_idx, val_idx

def custom_cv_subj(data):
    # custom cross-validation rule, each fold = 1 subject
    n, _ = data.shape
    subjects = 4
    # some actions are performed several times
    actions = 15
    for i in range(subjects):
        val_idx = np.arange(i * actions, i * actions + actions)
        train_idx = [x for x in range(subjects * actions) if x not in val_idx]
        yield train_idx, val_idx


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    # # ./experiments/cad60/svm01 for example
    # experiment_path =  os.path.join(args.model_dir, 'experiments', args.dataset_name, args.model_name + args.num)
    # if not os.path.isdir(experiment_path):
    #     os.makedirs(experiment_path)

    # json_file = os.path.join(experiment_path,'params.json')
    # if not os.path.isfile(json_file):
    #     raise ValueError(f'put json with params in {experiment_path}')

    # # for example ./experiments/cad60/svm01/params.json <- with this json file object Params is created 
    params = utils.Params()

    params.dataset_dir = args.dataset_dir
    params.environment = args.environment

    # HERE dataset name is saved from the parser
    params.dataset_name = args.dataset_name
    # params.model_version = args.model_name
    # params.experiment_path = experiment_path
    # params.mode = args.mode

    # specified in json
    # if params.gpu_id >= -1:
    #     params.cuda = True

    # Set the random seed for reproducible experiments
    np.random.seed(0)
    random.seed(0)

    # get the data
    dataset = fetch_data(params)
    X = dataset.data
    Y = dataset.label
    num_frames = dataset.valid_frame_num

    X = frame_by_frame_samples(extract_features_type1(X, num_frames), num_frames)
    Y = frame_by_frame_labels(Y, num_frames)

    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    custom_cv = custom_cv_subj_fbf(num_frames)
    scores = cross_val_score(clf, X, Y, cv=custom_cv)
    print(f'environment {params.environment}')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # load train data


