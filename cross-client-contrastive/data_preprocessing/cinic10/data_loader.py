import logging
import os

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import ImageFilter

from .datasets import ImageFolderTruncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cinic10_clean():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
                lambda x: F.pad(x.unsqueeze(0),
                                (4, 4, 4, 4),
                                mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
                lambda x: F.pad(x.unsqueeze(0),
                                (4, 4, 4, 4),
                                mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def _data_transforms_cinic10():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    augmentation = [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(x.unsqueeze(0),
                                (4, 4, 4, 4),
                                mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR_MEAN,  std=CIFAR_STD) ]

    trans = TwoCropsTransform(transforms.Compose(augmentation))
    
    return trans, trans


def load_cinic10_data(datadir):
    _train_dir = datadir + str('/train')
    logging.info("_train_dir = " + str(_train_dir))
    _test_dir = datadir + str('/test')
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ]))

    testset = ImageFolderTruncated(_test_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize(mean=cinic_mean,
                                                                                                 std=cinic_std),
                                                                            ]))
    X_train, y_train = trainset.imgs, trainset.targets
    X_test, y_test = testset.imgs, testset.targets
    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    X_train, y_train, X_test, y_test = load_cinic10_data(datadir)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    n_train = len(X_train)
    # n_test = len(X_test)

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CINIC10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CINIC10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs, transform=transform_train)
    test_ds = dl_obj(valdir, transform=transform_train)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def get_dataloader_cinic10_clean(datadir, train_bs):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10_clean()

    traindir = os.path.join(datadir, 'train')

    train_ds = dl_obj(traindir, transform=transform_train, returnindex = 1)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=False, drop_last=False)

    return train_dl


def load_partition_data_distributed_cinic10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
        test_data_num = len(test_data_global)
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                           dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        test_data_num = 0
        train_data_global = None
        test_data_global = None

    return train_data_num, test_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cinic10_1(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(train_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                           dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_partition_data_cinic10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, samplefile=None):
    BB = batch_size
    if samplefile is not None:
        _train_dir = os.path.join(data_dir, 'train')
        _test_dir = os.path.join(data_dir, 'train')
        transform_train, transform_test = _data_transforms_cinic10()
        al_samples = np.load(samplefile, allow_pickle=True)
        print('alsamples:', al_samples.shape)
        train_dataset = ImageFolderTruncated(_train_dir,transform=transform_train)
        test_dataset = ImageFolderTruncated(_test_dir, transform=transform_test)
        test_data_num = test_dataset.__len__()
        train_data_num = train_dataset.__len__()
        train_data_global = train_dataset.imgs
        test_data_global = test_dataset.imgs
        y_train = train_dataset.targets
        class_num = len(np.unique(y_train))
        nums = 0
        for i in al_samples:
            nums = nums + i.shape[0]
        logging.info("train_dl_global number = " + str(nums))

        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(client_number):
            samples_idx = al_samples[client_idx]
            #print(samples_idx)
            local_data_num = len(samples_idx)
            '''
            if local_data_num < 200:
                batch_size = int(BB/4)
            elif local_data_num < 400:
                batch_size = int(BB/2)
            else:
                batch_size = BB
            '''
            data_local_num_dict[client_idx] = local_data_num
            logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

            # training batch size = 64; algorithms batch size = 32
            
            if client_idx == 0:
                train_data_local = data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(samples_idx), drop_last=True)
                test_data_local = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

                test_data_local1 = test_data_local
            else:
                train_data_local = data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(samples_idx),  drop_last=True)
            
            #train_data_local, test_data_local1 = get_dataloader(dataset, data_dir, batch_size, batch_size)
            logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_idx, len(train_data_local), len(test_data_local1)))
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local1
        return train_data_num, test_data_num, train_data_global, test_data_global, \
               data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

        


    else:

        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                                data_dir,
                                                                                                partition_method,
                                                                                                client_number,
                                                                                                partition_alpha)
        class_num = len(np.unique(y_train))
        logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
        train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(train_data_global)))
        test_data_num = len(test_data_global)

        # get local dataset
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_idx in range(client_number):
            dataidxs = net_dataidx_map[client_idx]
            local_data_num = len(dataidxs)
            data_local_num_dict[client_idx] = local_data_num
            logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                            dataidxs)
            logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_idx, len(train_data_local), len(test_data_local)))
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local
        return train_data_num, test_data_num, train_data_global, test_data_global, \
            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
