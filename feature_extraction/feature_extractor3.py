from feature_extraction import tools
# import tools
from feature_extraction import features_lib as fl
import numpy as np

def extract_features_type3(X, num_frames):
	'''
	assumes data is already centered at torso
	in_shape = (N, C, T, V)
	out_shape = (N, num_features, T)
	''' 
	# N - samples, C - xyz, T - frames(all 2K for now, use num_frames to get actual length of a sample), V - joints 
	N, C, T, V = X.shape
	# we will use 14 features this time
	features = np.zeros((N, 14, T))
	# dist between two hands 11, 12 and a head 0
	features[:, 0, :] = tools.normalize_by_height(tools.dist_to_joint_allsamples(
    X[:, :, :, 11], X[:, :, :, 0], num_frames), X, num_frames)
    features[:, 1, :] = tools.normalize_by_height(tools.dist_to_joint_allsamples(
    X[:, :, :, 12], X[:, :, :, 0], num_frames), X, num_frames)
    # dist between elbows and head 4, 6
    features[:, 2, :] = tools.normalize_by_height(tools.dist_to_joint_allsamples(
    X[:, :, :, 4], X[:, :, :, 0], num_frames), X, num_frames)
    features[:, 3, :] = tools.normalize_by_height(tools.dist_to_joint_allsamples(
    X[:, :, :, 6], X[:, :, :, 0], num_frames), X, num_frames)
    # dist between two hands 11, 12
	features[:, 4, :] = tools.dist_to_joint_allsamples(X[:, :, :, 11], X[:, :, :, 12], num_frames)
	# torso inclination
	features[:, 5, :] = fl.body_incline(X, num_frames)
	# knee bend
	features[:, 6, :] = fl.knee_bend(X, num_frames)
	# head tilt
	features[:, 7, :] = fl.head_tilt(X, num_frames)
	# cos dist between elbows and hand
	features[:, 8, :] = 
    # temporal positional features

	# left hand 11 x, y
	features[:, 4, :] = tools.diff_position_x(X[:, :, :, 11], num_frames)
	features[:, 5, :] = tools.diff_position_y(X[:, :, :, 11], num_frames)
	# right hand 12
	features[:, 6, :] = tools.diff_position_x(X[:, :, :, 12], num_frames)
	features[:, 7, :] = tools.diff_position_y(X[:, :, :, 12], num_frames)
	# left elbow 4
	features[:, 8, :] = tools.diff_position_x(X[:, :, :, 4], num_frames)
	features[:, 9, :] = tools.diff_position_y(X[:, :, :, 4], num_frames)
	# right elbow 6
	features[:, 10, :] = tools.diff_position_x(X[:, :, :, 6], num_frames)
	features[:, 11, :] = tools.diff_position_y(X[:, :, :, 6], num_frames)
	# head 0
	features[:, 12, :] = tools.diff_position_x(X[:, :, :, 0], num_frames)
	features[:, 13, :] = tools.diff_position_y(X[:, :, :, 0], num_frames)
	# normalize and round
	# features = tools.normalize(features, num_frames)
	features = np.around(features, decimals=3)

	return features


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