
import argparse
import os
import random
import numpy as np

# for SVM
from sklearn import svm
from sklearn.model_selection import cross_val_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data0/', help="directory of the dataset")

# HERE it is decided which dataset will be used
parser.add_argument('--dataset_name', default='CAD-60', help="dataset name ")
parser.add_argument('--model_dir', default='./',
                    help="parents directory of model")

parser.add_argument('--model_name', default='SVM',help="model name")
# parser.add_argument('--load_model',
#         help='Optional, load trained models')
# parser.add_argument('--load',
#         type=str2bool,
#         default=False,
#         help='load a trained model or not ')
parser.add_argument('--mode', default='train', help='train,test,or load_train')
parser.add_argument('--num', default='01', help='num of trials (type: list)')


def fetch_data(params):
    # creates and returns feeder object with data
    if 'CAD-60' in params.dataset_name:
        params.data_feeder_args["data_path"] = params.dataset_dir+'/CAD-60'+'/train_data.npy'
        params.data_feeder_args["num_frame_path"] = params.dataset_dir+'/CAD-60'+'/train_num_frame.npy'
        params.data_feeder_args["label_path"] = params.dataset_dir + '/CAD-60' + '/train_label.pkl'
        dataset = Feeder(**params.data_feeder_args)
        return dataset
    else:
        raise NotImplementedError('only CAD-60 is supported')

def custom_cv_subj(data):
    # custom cross-validation rule, each fold = 1 subject
    n = data.N
    subjects = 4
    actions = 15
    for i in range(subjects):
        train_idx = np.arange(i * actions, i * actions + actions)
        val_idx = x for x in range(subjects * actions) if x not in train_idx
        yield train_idx, val_idx

# work in progress, not used for now
def gen_features(data):
    # xyz, frame, joint
    C, T, V = data.shape
    frames_selected = [5, 9, 14, 20, 27, 35, 44, 54, 65]

    # center at torso
    for t in range(T):
        # coords of the torso
        torso_coord = data[:, t, 2]
        for v in range(V):
        data_numpy[:, t, v] -= torso_coord

    # removing one dimension
    twodim_table = []
    for c in range(C):
        jointframe = []
        for t in range(T):
            if t in frames_selected:
                coords = []
                for v in range(V):
                    if v == 2:
                        continue
                    coords.append(data[c, t, v])
                np.hstack(jointframe, coords)
        twodim_table[c] = jointframe

    # removing another dimension
    onedim_table = []
    for i in  
    dist_euclid = np.linalg.norm(data[:. t, v])
                    distances.append(dist_euclid)
    return data


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    # ./experiments/cad60/svm01 for example
    experiment_path =  os.path.join(args.model_dir,'experiments',args.dataset_name,args.model_name+args.num)
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)


    json_file = os.path.join(experiment_path,'params.json')
    if not os.path.isfile(json_file):
        raise ValueError(f'put json with params in {experiment_path}')

    # for example ./experiments/cad60/svm01/params.json <- with this json file object Params is created 
    params = utils.Params(json_file)

    params.dataset_dir = args.dataset_dir

    # HERE dataset name is saved from the parser
    params.dataset_name = args.dataset_name
    params.model_version = args.model_name
    params.experiment_path = experiment_path
    params.mode = args.mode

    # specified in json
    # if params.gpu_id >= -1:
    #     params.cuda = True

    # Set the random seed for reproducible experiments
    np.random.seed(params.seed)
    random.seed(params.seed)

    # get the data
    dataset = fetch_data(params)
    data = dataset.data
    labels = dataset.labels

    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    custom_cv = custom_cv_subj()
    scores = cross_val_score(clf, data, labels, cv=custom_cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # load train data


