import numpy as np
from component_code.feature_extraction import extract_features

import pickle

_CLASS_NAMES = ['talking on the phone', 'writing on whiteboard', 'drinking water', 'rinsing mouth with water', 'brushing teeth', 'wearing contact lenses',
 'talking on couch', 'relaxing on couch', 'cooking (chopping)', 'cooking (stirring)', 'opening pill container', 'working on computer']
_MODEL_DIR = './models/all_final_model.pkl'


def predict_sample(sample):
    '''
    expects numpy array as input with shape (C, T, V), where C = 3 (x, y, z coords), T = 4 (number of frames) and V = 15, number of joints
    outputs 1 of 12 classes of activities
    '''
    with open(_MODEL_DIR, 'rb') as f:
        clf = pickle.load(f)

    x = extract_features(sample)
    y_pred = clf.predict_proba(x.reshape(1, -1))
    sorted_ind = np.argsort(y_pred[0])
    top_ind = np.flip(sorted_ind[-5:])
    top_probs = np.around(y_pred[0, top_ind], decimals=4)
    for i in range(len(top_ind)):
        print(_CLASS_NAMES[top_ind[i]] + ': ' + str(top_probs[i]))


