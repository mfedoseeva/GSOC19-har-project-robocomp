import numpy as np

def center(X):
    '''
    origin is moved to torso coordinates for each frame
    '''
    # c - xyz, t - frames, v - join
    C, T, V = X.shape
    for t in range(T):
        # torso is the 3rd joint
        torso_coord = X[:, t, 2]
        for v in range(V):
            X[:, t, v] -= torso_coord
    return X

def dist_to_torso(joint_data):
    '''
    torso coords are always zero since we center relative to torso.
    in_shape = (3, n_frames)
    out_shape = (n_frames)
    '''
    C, T = joint_data.shape
    dist_data = np.zeros(T)
    for t in range(T):
        dist_data[t] = np.linalg.norm(joint_data[:, t])   
    return dist_data

def dist_to_torso_single(joint_data):
    '''
    dist_to_torso for single frame
    '''
    return np.linalg.norm(joint_data)   

def dist_to_joint(joint1, joint2):
    '''
    in_shape = (3, n_frames) for both
    out_shape = (n_frames)
    '''
    C_1, T_1 = joint1.shape
    C_2, T_2 = joint2.shape
    assert(C_1 == C_2 and T_1 == T_2)
    dist_data = np.zeros(T_1)
    for t in range(T_1):
        dist_data[t] = np.linalg.norm(joint1[:, t] - joint2[:, t])
    return dist_data

def dist_to_joint_single(joint1, joint2):
    '''
    dist_to_torso for single frame
    '''
    assert(len(joint1) == len(joint2))
    return np.linalg.norm(joint1 - joint2)

def flatten(X):
    '''
    turn two-dim data into one-dim data by stacking
    in_shape = (num_frames, features)
    out_shape = (num_frames*features)
    '''
    T, V = X.shape
    data = np.zeros(T * V)
    for i in range(T):
        data[i*T : i*T + T] = X[i]
    return data

def cos_dist_torso(joint1, joint2):
    '''
    cosine distance of a joint relative to another joint
    in_shape = (3, n_frames)
    out_shape = (3, n_frames)
    '''
    C_1, T_1 = joint1.shape
    C_2, T_2 = joint2.shape
    assert(C_1 == C_2 and T_1 == T_2)
    
    out = np.zeros(T_1)
    for t in range(T_1):
        dot_prod = np.dot(joint1[t], joint2[t])
        a = np.linalg.norm(joint1[:,t])
        b = np.linalg.norm(joint2[:, t])
        out[t] = dot_prod/(a * b)   
    return out

def horizontal_flip(X):
    '''
    flip horizontally all data relative to torso, assumes that data is centered at torso
    in_shape = out_shape
    '''
    C, T, V = X.shape
    for t in range(T):
        X[0, t, :] *= -1