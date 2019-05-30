
import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data0/', help="directory of the dataset")

# HERE it is decided which dataset will be used
parser.add_argument('--dataset_name', default='CAD-60', help="dataset name ")
parser.add_argument('--model_dir', default='./',
                    help="parents directory of model")

parser.add_argument('--model_name', default='SVM',help="model name")
parser.add_argument('--load_model',
        help='Optional, load trained models')
parser.add_argument('--load',
        type=str2bool,
        default=False,
        help='load a trained model or not ')
parser.add_argument('--mode', default='train', help='train,test,or load_train')
parser.add_argument('--num', default='01', help='num of trials (type: list)')



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
    if params.gpu_id >= -1:
        params.cuda = True

    # Set the random seed for reproducible experiments
    np.random.seed(params.seed)
    random.seed(params.seed)


    # instantiate nodel object, ** unpacks key-value pairs to the function

    if 'CAD' in params.model_version:
        model = model.model(**params.model_args)

        loss_fn = model.loss_fn

    # load train data


