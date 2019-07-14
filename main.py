import argparse
import os
import random
import numpy as np
from feature_extraction.feature_extractor_for_sequence import extract_features
from feature_extraction.tools import *
from feeder.feeder import Feeder
from feeder import utils
from support_operations.plot_confusion_matrix import plot_confusion_matrix
import pickle

# for SVM
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

# for ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

_ENVIRONMENT = ['office', 'livingroom', 'kitchen', 'bedroom', 'bathroom']
_SUBJECTS = 4
_CLASS_NAMES = ['talking on the phone', 'writing on whiteboard', 'drinking water', 'rinsing mouth with water', 'brushing teeth', 'wearing contact lenses', 'talking on couch', 'relaxing on couch', 'cooking (chopping)', 'cooking (stirring)', 'opening pill container', 'working on computer']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./data0/', help="directory of the dataset")

# HERE it is decided which dataset will be used
parser.add_argument('--dataset_name', default='CAD-60', help="dataset name ")
# this parameter controls if only cv accuracy is output or the model is fit and confusion matrices saved
parser.add_argument('--evaluation', default='cv', help='cv or full')


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

def custom_cv_subj(new_samples_num, augm=False, subjects=_SUBJECTS):
    ''' 
    custom cross-validation rule, each fold = 1 subject. augm flag is False if augmented data is not used for training
    '''
    n = len(new_samples_num)
    actions = int(n/subjects)
    
    samples_per_subject = np.zeros(subjects, dtype=int)
    
    # determine now many samples we have per subjects
    for i in range(subjects):
        samples_per_subject[i] = sum(new_samples_num[i * actions : i * actions + actions])

    subj_end_idx = [None]*subjects

    prev = 0
    for i in range(subjects):
        subj_end_idx[i] = samples_per_subject[i] + prev
        prev += samples_per_subject[i] 

    start = 0
    # we want to train on all samples except for the frames belonging to a subject that is used as validation and the augmented 
    # samples belonging to that subject being validated
    total_samples = sum(new_samples_num)
    for i in range(subjects):
        val_idx = np.arange(start, subj_end_idx[i])
        if augm:
            excluded_from_training = np.arange(total_samples + start, total_samples + subj_end_idx[i])
        start += samples_per_subject[i]
        if augm:
            train_idx = [x for x in range(2 * total_samples) if x not in val_idx and x not in excluded_from_training]
        else:
            train_idx = [x for x in range(total_samples) if x not in val_idx]
        yield train_idx, val_idx


def total_mean_acc(env_classified, total_acc):
    total_classified = sum(env_classified)
    acc = 0
    for i in range(len(total_acc)):
        acc += total_acc[i] * env_classified[i] / total_classified
    return np.around(acc, decimals=2)

def save_model(clf, environment):
    os.makedirs('./models', exist_ok=True)
    with open('./models/{}_final_model.pkl'.format(environment), 'wb') as f:
        pickle.dump(clf, f)

def print_distribution(labels, total_classes=12):
    # print classes distribution
    print('Distribution of samples across label bins: ')
    oneh_vector = np.zeros(12)
    bins = np.bincount(labels)
    for i in range(len(bins)):
        oneh_vector[i] = bins[i] 
    print(oneh_vector)
    env_classified.append(len(labels))

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    params = utils.Params()

    params.dataset_dir = args.dataset_dir
    params.evaluation = args.evaluation
    # HERE dataset name is saved from the parser
    params.dataset_name = args.dataset_name

    # Set the random seed for reproducible experiments
    np.random.seed(0)
    random.seed(0)

    env_classified = []
    total_acc = []

    for env in _ENVIRONMENT:

        print('')

        params.environment = env
        print(f'Environment: {params.environment}')

        # get the data
        dataset = fetch_data(params)
        X = dataset.data
        Y = dataset.label
        num_frames = dataset.valid_frame_num


        X = center(X, num_frames)

        X_augm = np.copy(X)
        X_augm = horizontal_flip(X_augm, num_frames)
        Y_orig = np.copy(Y)

        X, Y, new_samples_num = extract_features(X, Y, num_frames, seq_length=100, sampled_freq=25)
        X = normalize_allsamples(X)

        X_augm, _, _ = extract_features(X_augm, Y_orig, num_frames, seq_length=100, sampled_freq=25)
        X_augm = normalize_allsamples(X_augm)

        print('training data shape: ')
        print(X.shape)

        print_distribution(Y)
        

        X = np.vstack((X, X_augm))
        Y = np.concatenate((Y, Y), axis=None)

        # pca = PCA(n_components=12)
        # pca.fit(X)
        # print(pca.explained_variance_ratio_)  
        # print(pca.components_)

        clf = svm.SVC(decision_function_shape='ovo', gamma='scale')

        if params.evaluation == 'cv' or 'full':
            # runs cross validation and outputs accuracy
            custom_cv = custom_cv_subj(new_samples_num, augm=True)
            scores = cross_val_score(clf, X, Y, cv=custom_cv, n_jobs=-1)
            print('Accuracy for each fold, i.e. subject')
            for i in range(4):
                print(scores[i])
            print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            total_acc.append(scores.mean())
        if params.evaluation == 'full':
            # produces confusion matrices
            pred_labels = np.empty(int(len(Y)/2), dtype=int)
            correct_labels = np.empty(int(len(Y)/2), dtype=int)
            custom_cv = custom_cv_subj(new_samples_num, augm=True)
            prev = 0
            for i in custom_cv:
                train_idx, test_idx = i
                X_train = X[train_idx]
                X_test = X[test_idx]
                Y_train = np.array(Y)[train_idx]
                Y_test = np.array(Y)[test_idx]
                Y_pred = clf.fit(X_train, Y_train).predict(X_test)
                pred_len = len(Y_pred)
                pred_labels[prev : (prev + pred_len)] = Y_pred
                correct_labels[prev : (prev + pred_len)] = Y_test
                prev = prev + pred_len
            np.set_printoptions(precision=2)
            plot_confusion_matrix(correct_labels, pred_labels, classes=np.array(_CLASS_NAMES), normalize=True, title=env)
    print('')
    print('Total average accuracy across all environments: ')
    print(total_mean_acc(env_classified, total_acc))

