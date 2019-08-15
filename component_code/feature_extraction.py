import numpy as np
import pickle

# this values were used during training
_SAMPLED_FREQ = 30
_SEQ_LENGTH = 120
_NUM_FEATURES = 16
# this file contains values to normalize the test sample
_NORMAL_DENOMS = './normalization_denoms.pkl'

def extract_features_full_frames(sample):
    '''
    this code extracts features from one sample of 120 frames at inference time
    '''
    x = center(sample)
    x = cut_sequence(x, length=_SEQ_LENGTH)
    # this operation will produce 4 samples, since length of sequence is 120. We will run inference on all of them and then pick the majority prediction
    X = sample_from_sequence(x, freq=_SAMPLED_FREQ)
    N_new, _, T_new, _ = X.shape

    features = np.zeros((N_new, _NUM_FEATURES, T_new))

    for i in range(N_new):
        # dist between two hands 11, 12 and torso
        features[i, 0, :] = dist_to_joint(X[i, :, :, 11], X[i, :, :, 2])
        features[i, 1, :] = dist_to_joint(X[i, :, :, 12], X[i, :, :, 2])
        # dist between elbows and torso
        features[i, 2, :] = dist_to_joint(X[i, :, :, 4], X[i, :, :, 2])
        features[i, 3, :] = dist_to_joint(X[i, :, :, 6], X[i, :, :, 2])
        # dist between two hands 11, 12 
        features[i, 4, :] = dist_to_joint(X[i, :, :, 11], X[i, :, :, 12])
        # dist between head and torso
        features[i, 5, :] = dist_to_joint(X[i, :, :, 0], X[i, :, :, 2])
        # dist between shoulders and feet 3, 5, 13, 14 
        features[i, 6, :] = dist_to_joint(
            X[i, :, :, 3], X[i, :, :, 13]) + dist_to_joint(
            X[i, :, :, 5], X[i, :, :, 14])
        # knees to torso
        features[i, 7, :] = dist_to_joint(X[i, :, :, 8], X[i, :, :, 2]) + dist_to_joint(
            X[i, :, :, 10], X[i, :, :, 2])

        # temporal positional features
        # left hand 11 x, y
        features[i, 8, :] = diff_position_x(X[i, :, :, 11])
        features[i, 9, :] = diff_position_y(X[i, :, :, 11])
        # right hand 12
        features[i, 10, :] = diff_position_x(X[i, :, :, 12])
        features[i, 11, :] = diff_position_y(X[i, :, :, 12])
        # head 0
        features[i, 12, :] = diff_position_x(X[i, :, :, 0])
        features[i, 13, :] = diff_position_y(X[i, :, :, 0])
        # to understand if we are viewing from the side
        features[i, 14, :] = body_turn(X[i, :, :, :])
        # hands to knees
        features[i, 15, :] = dist_to_joint(X[i, :, :, 11], X[i, :, :, 8]) + dist_to_joint(
            X[i, :, :, 12], X[i, :, :, 10])
    # round
    features = np.around(features, decimals=4)
    flat_features = flatten(features)

    final_features = np.zeros(flat_features.shape)
    for i in range(flat_features.shape[0]):
        final_features[i] = normalize(flat_features[i])
    return final_features

def extract_features_four_frames(sample):
    '''
    extracts features for 4 frames which are alreadt sampled 
    '''
    x = center(sample)
    T = x.shape[1]
    features = np.zeros((_NUM_FEATURES, T))
    
    features[0, :] = dist_to_joint(x[:, :, 11], x[:, :, 2])
    features[1, :] = dist_to_joint(x[:, :, 12], x[:, :, 2])
    # dist between elbows and torso
    features[2, :] = dist_to_joint(x[:, :, 4], x[:, :, 2])
    features[3, :] = dist_to_joint(x[:, :, 6], x[:, :, 2])
    # dist between two hands 11, 12 
    features[4, :] = dist_to_joint(x[:, :, 11], x[:, :, 12])
    # dist between head and torso
    features[5, :] = dist_to_joint(x[:, :, 0], x[:, :, 2])
    # dist between shoulders and feet 3, 5, 13, 14 
    features[6, :] = dist_to_joint(
        x[:, :, 3], x[:, :, 13]) + dist_to_joint(
        x[:, :, 5], x[:, :, 14])
    # knees to torso
    features[7, :] = dist_to_joint(x[:, :, 8], x[:, :, 2]) + dist_to_joint(
        x[:, :, 10], x[:, :, 2])

    # temporal positional features
    # left hand 11 x, y
    features[8, :] = diff_position_x(x[:, :, 11])
    features[9, :] = diff_position_y(x[:, :, 11])
    # right hand 12
    features[10, :] = diff_position_x(x[:, :, 12])
    features[11, :] = diff_position_y(x[:, :, 12])
    # head 0
    features[12, :] = diff_position_x(x[:, :, 0])
    features[13, :] = diff_position_y(x[:, :, 0])
    # to understand if we are viewing from the side
    features[14, :] = body_turn(x[:, :, :])
    # hands to knees
    features[15, :] = dist_to_joint(x[:, :, 11], x[:, :, 8]) + dist_to_joint(
        x[:, :, 12], x[:, :, 10])    

    features = np.around(features, decimals=4)
    flat_features = features.flatten()
    final_features = normalize(flat_features)
    return final_features



def center(x):
    '''
    mode origin to torso, joint 2
    '''
    C, T, V = x.shape
    for t in range(T):
        torso_coord = x[:, t, 2]
        for v in range(V):
            x[:, t, v] -= torso_coord
    return x

def cut_sequence(x, length):
    '''
    cuts the sample down to 'length', if frames are not sufficient, will raise an error; if frames are too many will samply discard the excess
    '''
    if x.shape[1] < _SEQ_LENGTH:
        raise ValueError('the data sequence is too short, there should be at least 120 frames')
    return x[:, :length, :]

def sample_from_sequence(x, freq):
    '''
    '''
    C, T, V = x.shape

    idx = []
    for i in range(freq):
        idx.append([x + i for x in range(T//freq*freq) if x%freq == 0])
    
    X_reduced = np.zeros((freq, C, T//freq, V))
    for j in range(freq):
        for k, ind in enumerate(idx[j]):
            X_reduced[j, :, k, :] = x[:, ind, :]
    return X_reduced

def dist_to_joint(joint1, joint2):
    '''
    in_shape = (3, T) for both
    out_shape = (T)
    '''
    _, T = joint1.shape
    dist_data = np.zeros(T)
    for t in range(T):
        dist_data[t] = np.linalg.norm(joint1[:, t] - joint2[:, t])
    return dist_data

def diff_position_x(x):
    '''
    in_shape of joint (3, T)
    out_shape (T)
    '''
    _, T = x.shape
    out = np.zeros(T)
    init_x = x[0, 0]
    for t in range(T):
        out[t] = init_x - x[0, t]
        # prevent zero values, as in the end we get that for all first frames we have 0 and it will result in NaN when normalizing
        if out[t] < abs(1e-3):
            out[t] = 1e-3
    return out

def diff_position_y(x):
    '''
    in_shape of joint (3, T)
    out_shape (T)
    '''
    _, T = x.shape
    out = np.zeros(T)
    init_y = x[1, 0]
    for t in range(T):
        out[t] = init_y - x[1, t]
        if out[t] < abs(1e-3):
            out[t] = 1e-3
    return out

def body_turn(x):
    '''
    calculates the body turn by calculating the distance between the left and the rights shoulder based on 2d coordinates and divides by the
    distance between the same joints based on 3d coordinates
    in_shape = (C, T, V)
    out_shape = (T)
    '''
    C, T, V = x.shape
    out = np.zeros(T)
    for t in range(T):
        l_should_2d = x[0:2, t, 3]
        r_should_2d = x[0:2, t, 5]
        dist_2d = dist_to_joint_single(l_should_2d, r_should_2d)
        out[t] = dist_2d/dist_to_joint_single(x[:, t, 3], x[:, t, 5]) 
    return out

def dist_to_joint_single(joint1, joint2):
    '''
    dist_to_torso for single frame
    '''
    return np.linalg.norm(joint1 - joint2)

def flatten(features):
    '''
    this method is for features of shape = (N, feat, T) to transform them into
    shape (N, feat*T) by means of horizontal stacking
    '''
    N, feat, T = features.shape
    new_features = np.zeros((N, feat * T))
    for i in range(N):
        new_features[i, :] = np.hstack(features[i, :, :])
    return new_features


def normalize(features):
    '''
    we will normalize using the coefficients obtained during training to have the similar features
    '''
    with open(_NORMAL_DENOMS, 'rb') as f:
        denominators = pickle.load(f)
    F = features.shape[0]
    # if cutting length, sampling frequency and number of features have not been changed, the F should be the same as len of denominators
    assert(len(denominators) == F)
    for j in range(F):
        features[j] = features[j]/denominators[j]
    return features
