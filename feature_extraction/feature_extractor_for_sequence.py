from feature_extraction import tools
# import tools
from feature_extraction import features_lib as fl
import numpy as np

def extract_features(X, labels, num_frames, seq_length=75, sampled_freq=5):
    '''
    assumes data is already centered at torso
    in_shape = (N, C, T, V)
    out_shape = (N, num_features, T)
    ''' 
    # N - samples, C - xyz, T - frames(all 2K for now, use num_frames to get actual length of a sample), V - joints 
    N, C, T, V = X.shape

    X_cut, new_samples_num = tools.cut_samples(X, num_frames, seq_length)
    X_sampled, new_samples_num = tools.sample_from_cut_sequence(X_cut, new_samples_num, sampled_freq)
    labels_new = tools.labels_for_cut_samples(labels, new_samples_num)

    X = X_sampled
    # new X.shape is (N*valid_frame_num//seq_length*sampled_freq, C, seq_length//sampled_freq, V)
    # num_frames is the same for all samples now

    N_new, _, T_new, _ = X.shape
    num_frames = [T_new] * N_new

    features = np.zeros((N_new, 16, T_new))

    # dist between two hands 11, 12 and torso
    features[:, 0, :] = tools.dist_to_joint_allsamples(X[:, :, :, 11], X[:, :, :, 2], num_frames)
    features[:, 2, :] = tools.dist_to_joint_allsamples(X[:, :, :, 12], X[:, :, :, 2], num_frames)
    # dist between elbows and torso
    features[:, 2, :] = tools.dist_to_joint_allsamples(X[:, :, :, 4], X[:, :, :, 2], num_frames)
    features[:, 3, :] = tools.dist_to_joint_allsamples(X[:, :, :, 6], X[:, :, :, 2], num_frames)
    # dist between two hands 11, 12 - very important
    features[:, 4, :] = tools.dist_to_joint_allsamples(X[:, :, :, 11], X[:, :, :, 12], num_frames)
    # torso inclination - can improve bedroom performance if shoulders to feet is removed, but then all other envs drop. without hip and feet improves bathroom
    # drops a bit others
    # features[:, 4, :] = fl.body_incline(X, num_frames)
    # knee bend - worsens the performance on bedroom by a lot if used together with hip to feet, without hip to feet improves significantly
    # living room but worsens significantly the bedroom
    # features[:, 4, :] = fl.knee_bend(X, num_frames)
    # head tilt
    # features[:, 4, :] = fl.head_tilt(X, num_frames)

    # dist between head and torso
    features[:, 5, :] = tools.dist_to_joint_allsamples(X[:, :, :, 0], X[:, :, :, 2], num_frames)
    # dist between shoulders and feet 3, 5, 13, 14 - important
    features[:, 6, :] = tools.dist_to_joint_allsamples(
        X[:, :, :, 3], X[:, :, :, 13], num_frames) + tools.dist_to_joint_allsamples(
        X[:, :, :, 5], X[:, :, :, 14], num_frames)
    # dist between shoulders and hands
    # features[:, 6, :] = tools.dist_to_joint_allsamples(X[:, :, :, 3], X[:, :, :, 11], num_frames) + tools.dist_to_joint_allsamples(
        # X[:, :, :, 5], X[:, :, :, 12], num_frames)
    # dist between hip and feet 7, 9, 13, 14 - drops the performance
    # features[:, 6, :] = tools.dist_to_joint_allsamples(
        # X[:, :, :, 7], X[:, :, :, 13], num_frames) + tools.dist_to_joint_allsamples(X[:, :, :, 9], X[:, :, :, 14], num_frames)

    # knees to torso
    features[:, 7, :] = tools.dist_to_joint_allsamples(X[:, :, :, 8], X[:, :, :, 2], num_frames) + tools.dist_to_joint_allsamples(
        X[:, :, :, 10], X[:, :, :, 2], num_frames)
    # cos dist features - drop the performance a lot
    # left_elbow_hand = tools.cos_dist2(X[:, :, :, 3], X[:, :, :, 11], num_frames)
    # right_elbow_hand = tools.cos_dist2(X[:, :, :,5], X[:, :, :, 12], num_frames)
    # features[:, 6, :] = left_elbow_hand
    # features[:, 7, :] = right_elbow_hand

    # temporal positional features
    # left hand 11 x, y
    features[:, 8, :] = tools.diff_position_x(X[:, :, :, 11], num_frames)
    features[:, 9, :] = tools.diff_position_y(X[:, :, :, 11], num_frames)
    # right hand 12
    features[:, 10, :] = tools.diff_position_x(X[:, :, :, 12], num_frames)
    features[:, 11, :] = tools.diff_position_y(X[:, :, :, 12], num_frames)
    # # left elbow 4
    # features[:, 12, :] = tools.diff_position_x(X[:, :, :, 4], num_frames)
    # features[:, 13, :] = tools.diff_position_y(X[:, :, :, 4], num_frames)
    # # right elbow 6
    # features[:, 14, :] = tools.diff_position_x(X[:, :, :, 6], num_frames)
    # features[:, 15, :] = tools.diff_position_y(X[:, :, :, 6], num_frames)
    # head 0
    features[:, 12, :] = tools.diff_position_x(X[:, :, :, 0], num_frames)
    features[:, 13, :] = tools.diff_position_y(X[:, :, :, 0], num_frames)
    # if we are vieweing from the side
    features[:, 14, :] = fl.body_turn(X, num_frames)
    # hands to knees
    features[:, 15, :] = tools.dist_to_joint_allsamples(X[:, :, :, 11], X[:, :, :, 8], num_frames) + tools.dist_to_joint_allsamples(
        X[:, :, :, 12], X[:, :, :, 10], num_frames)
    # round
    features = np.around(features, decimals=4)

    flat_features = tools.flatten(features)

    return flat_features, labels_new, new_samples_num


if __name__ == '__main__':

    import sys
    sys.path.insert(0, '../feeder')
    from feeder import Feeder

    # a simple test

    data_path = "../data0/CAD-60/office/train_data.npy"
    label_path = "../data0/CAD-60/office/train_label.pkl"
    num_frame_path = "../data0/CAD-60/office/train_num_frame.npy"
    dataset = Feeder(data_path, label_path, num_frame_path)

    X = dataset.data
    num_frame = dataset.valid_frame_num

    features = extract_features_type1(X, num_frame)

    print(features[15, :, :10])