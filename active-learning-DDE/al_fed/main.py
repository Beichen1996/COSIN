import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb


from fedavg.fedavg_api import FedAvgAPI
from fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from data_preprocessing.cifar10.data_loader import load_partition_data_cifar10_al
from data_preprocessing.cifar100.data_loader import load_partition_data_cifar100_al
from data_preprocessing.cinic10.data_loader import load_partition_data_cinic10_al
from model.cv.mobilenet import mobilenet
from model.cv.resnet import resnet50
from model.cv.cnn import CNN_DropOut
from model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow


from model.linear.lr import LogisticRegression

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# python  main_fedavg.py --model=resnet56 --dataset=cifar10 --data_dir='/data1/beichen_zhang/FedML/data/cifar10' --partition_method=hetero --batch_size=128 --client_optimizer=sgd --lr=0.001 --epochs=1 --client_num_in_total=20 --client_num_per_round=10 --comm_round=200  --gpu=1

RATE = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet50', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--num', type=int, default=50000, 
                        help='number of samples in data pool')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--sample_dir', type=str, default='./cifar10',
                        help='sample index directory')

    parser.add_argument('--version', type=str, default='0',
                        help='initial sampling version')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0005)

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=1000,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    return parser


def load_data(args, dataset_name,samplefile = None):
    # check if the centralized training is enabled

    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10_al
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100_al
    elif dataset_name == "cinic10":
        data_loader = load_partition_data_cinic10_al
    else:
        data_loader = load_partition_data_cifar10_al
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_num_in_total, args.batch_size, samplefile)


    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" :
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "resnet50":
        model = resnet50(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    return model


def custom_model_trainer(args, model):
    return MyModelTrainerCLS(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name="FedAVG-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # load data


    for rate in RATE:
        sample_num = int(rate * args.num / args.client_num_in_total)
        samplefile = args.sample_dir + '/' + str(sample_num) + '_' + str(args.client_num_in_total) + '_' + args.partition_method +'_v' + args.version +'.npy'
        accfile = open('log/' + args.dataset + '_' + str(sample_num) + '_' + str(args.client_num_in_total) + '_' + args.partition_method +'_v' + args.version + '.txt', 'a')
        
        logging.info(samplefile)

        dataset = load_data(args, args.dataset, samplefile = samplefile)
        model = create_model(args, model_name=args.model, output_dim=dataset[7])
        print('classnum:' , dataset[7])
        model_trainer = custom_model_trainer(args, model)
        fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, acclogfile = accfile)


        fedavgAPI.train()
        accfile.close()

    #fedavgAPI.train()
