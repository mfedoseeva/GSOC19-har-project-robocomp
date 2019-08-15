import numpy as np
from component_code.feature_extraction import extract_features_full_frames
from component_code.feature_extraction import extract_features_four_frames

import pickle

_CLASS_NAMES = ['talking on the phone', 'writing on whiteboard', 'drinking water', 'rinsing mouth with water', 'brushing teeth', 'wearing contact lenses',
 'talking on couch', 'relaxing on couch', 'cooking (chopping)', 'cooking (stirring)', 'opening pill container', 'working on computer']
_MODEL_DIR = './models/all_final_model.pkl'


def predict_sample(sample):
    '''
    expects numpy array as input with shape (C, T, V), where C = 3 (x, y, z), T = 4, number of frames and V = 15, number of joints
    outputs 1 of 12 classes of activities
    '''
    with open(_MODEL_DIR, 'rb') as f:
        clf = pickle.load(f)

    # commented part is alternative where T = 120
    # extracting features will return 30 samples, we will run inference on all of them and select the majority vote
    # X = extract_features_full_frames(sample)
    # print(X.shape)
    # Y_pred = clf.predict(X)
    # final_pred = np.bincount(Y_pred).argmax()
    # label = _CLASS_NAMES[final_pred]
    # print(Y_pred)
    # print(final_pred)
    # print(label)

    # assuming sample is already 4 cut frames
    x = extract_features_four_frames(sample)
    y_pred = clf.predict_proba(x.reshape(1, -1))
    top_ind = np.argsort(y_pred[0])
    top_ind = np.flip(top_ind[-5:])
    top_probs = np.around(y_pred[0, top_ind], decimals=4)
    for i in range(len(top_ind)):
        print(_CLASS_NAMES[top_ind[i]] + ': ' + str(top_probs[i]))


