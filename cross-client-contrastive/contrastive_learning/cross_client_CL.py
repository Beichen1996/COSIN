import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as transforms


from utils.fedavg_api import FedAvgAPI
from utils.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from model.cv.mobilenet import mobilenet
from model.cv.resnet import resnet50
from model.cv.cnn import CNN_DropOut

from model.linear.lr import LogisticRegression
from model.cv.resnet_gn import resnet18
from utils.client import MoCo



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet50', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dim', type=int, default=128, 
                        help='dimension of the encoder')
    
    parser.add_argument('--K', type=int, default=2000, 
                        help='dimension of the encoder')


    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=1e-4)

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False


    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100
    elif dataset_name == "cinic10":
        data_loader = load_partition_data_cinic10
    else:
        data_loader = load_partition_data_cifar10
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_num_in_total, args.batch_size, None)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

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
    if model_name == "lr" and args.dataset == "mnist":
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


    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # load data
    
    modelq = create_model(args, model_name=args.model, output_dim=128)
    modelk = create_model(args, model_name=args.model, output_dim=128)
    #modelq.load_state_dict(torch.load('encoder_' + args.dataset + '.pth'))
    model = MoCo(modelq, modelk, dim=128, K=6400, m=0.999, T=0.07, device = device)

    dataset = load_data(args, args.dataset)
    
    
    model_trainer = custom_model_trainer(args, model)

    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
    fedavgAPI.train()

    if not os.path.exists(args.dataset):  # 是否存在这个文件夹
        os.makedirs(args.dataset)
    torch.save(modelk.state_dict(), args.dataset + '/encoder_' + args.dataset + '.pth')


