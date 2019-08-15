from feeder.feeder import Dataset
from component_code.inference import predict_sample
from feeder import utils

_CLASS_NAMES = ['talking on the phone', 'writing on whiteboard', 'drinking water', 'rinsing mouth with water', 'brushing teeth', 'wearing contact lenses',
 'talking on couch', 'relaxing on couch', 'cooking (chopping)', 'cooking (stirring)', 'opening pill container', 'working on computer']

if __name__ == '__main__':
    params = utils.Params()
    params.data_feeder_args["data_path"] = './data0/CAD-60/all/train_data.npy'
    params.data_feeder_args["num_frame_path"] = './data0/CAD-60/all/train_num_frame.npy'
    params.data_feeder_args["label_path"] = './data0/CAD-60/all/train_label.pkl'
    dataset = Dataset(**params.data_feeder_args)

    X = dataset.data
    Y = dataset.label
    x = X[10, :, :, :]
    y = Y[10]
    idx = [1, 30, 60, 90]
    x = x[:, idx, :]
    print('true label: ')
    print(_CLASS_NAMES[y])
    print('prediction top 5: ')
    predict_sample(x)