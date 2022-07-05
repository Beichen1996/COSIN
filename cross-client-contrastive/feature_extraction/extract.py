import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as transforms



sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from data_preprocessing.cifar10.data_loader import get_dataloader_CIFAR10_clean
from data_preprocessing.cifar100.data_loader import get_dataloader_CIFAR100_clean
from data_preprocessing.cinic10.data_loader import get_dataloader_cinic10_clean
from model.cv.mobilenet import mobilenet
from model.cv.resnet50 import resnet50
from model.cv.cnn import CNN_DropOut

from model.linear.lr import LogisticRegression
#from client import MoCo



def add_args(parser):
    parser.add_argument('--model', type=str, default='resnet50', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dim', type=int, default=128, 
                        help='dimension of the encoder')
    
    parser.add_argument('--K', type=int, default=2000, 
                        help='dimension of the encoder')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    
    parser.add_argument('--pretrained', type=str, default='./pretrain/cifar10_contrastive.pth', metavar='N',
                        help='pretrained path')

    parser.add_argument('--data_dir', type=str, default='../../data/cifar10',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')


    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu')
    return parser


def load_data(args, dataset_name):

    if dataset_name == "cifar10":
        data_loader = get_dataloader_CIFAR10_clean
    elif dataset_name == "cifar100":
        data_loader = get_dataloader_CIFAR100_clean
    elif dataset_name == "cinic10":
        data_loader = get_dataloader_cinic10_clean
    else:
        data_loader = get_dataloader_CIFAR10_clean
    dataloader_ex = data_loader(args.data_dir, args.batch_size)
    return dataloader_ex



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
    
    model = create_model(args, model_name=args.model, output_dim=128)
    for name, param in model.named_parameters():
        param.requires_grad = False
    # init the fc layer
    dim_mlp = model.fc.weight.shape[1]
    model.fc = torch.nn.Sequential(torch.nn.Linear(dim_mlp, dim_mlp), torch.nn.ReLU(), model.fc) 

    '''
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') :
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    torch.save(model.state_dict(), 'cifar10_contrastive.pth')
    torch.save(model.state_dict(), 'cifar100_contrastive.pth')
    torch.save(model.state_dict(), 'cinic10_contrastive.pth')
    '''

    state_dict = torch.load(args.pretrained, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)


    dataloader_ex = load_data(args, args.dataset)

    indexs = []
    features = []
    #(images, _) = dataset[i]
    for i, (images, _, index) in enumerate(dataloader_ex):
        print(i)
        images = images.cuda(args.gpu)
        indexs.append(index)
        fs = model(images)
        fs = fs.cpu()
        features.append(fs)

    indexs = torch.cat(indexs,0)
    features = torch.cat(features,0)
    sortidx = torch.sort(indexs)[1]
    indexs = indexs[sortidx]
    features = features[sortidx]
    print(indexs.shape, indexs)
    print(features.shape)
    torch.save(features, args.dataset + "feature.pt")
