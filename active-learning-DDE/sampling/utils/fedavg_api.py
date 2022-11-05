import copy
import logging
import random

import numpy as np
import torch

from utils.client import Client

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer, acclogfile = None):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, train_unlabeled_data_local_dict] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.train_unlabeled_data_local_dict = train_unlabeled_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.acclog = acclogfile

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global, w2_global = self.model_trainer.get_model_params()
        bestacc = 0
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            w2_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w, w2 = client.train(  (copy.deepcopy(w_global), copy.deepcopy(w2_global))  )
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                w2_locals.append((client.get_sample_number(), copy.deepcopy(w2)))

            # update global weights
            w_global = self._aggregate(w_locals)
            w2_global = self._aggregate(w2_locals)
            self.model_trainer.set_model_params((w_global, w2_global))

            # test results
            # at last round
            
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                test_acc,test_loss = self._local_test_on_all_clients(round_idx)
                if test_acc > bestacc:
                    bestacc = test_acc
            


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(1):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            w_global = self.model_trainer.get_model_params()
            client.set_params(w_global)

            if self.test_data_local_dict[client_idx] is None:
                continue
            

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])


        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats)
        return test_acc,test_loss


    
    def sampling(self, labeled_idx, unlabeled_idx, device, budget):
        unlabeled_data = self.train_unlabeled_data_local_dict
        new_labeled_idx=[]
        new_unlabeled_idx=[]
        
        for client_idx in range(self.args.client_num_per_round):
            
            labeled_idx_local = labeled_idx[client_idx]
            unlabeled_idx_local = unlabeled_idx[client_idx]
            samplenum = int(budget*(len(labeled_idx_local) + len(unlabeled_idx_local)) )
            
            unlabeled_dataloader = unlabeled_data[client_idx]
            scores = self.model_trainer.get_uncertainty(unlabeled_dataloader, device)
            print('number of unlabeled: ', scores.shape, samplenum)
            arg = np.argsort(scores).numpy()

            new_labeled_set = unlabeled_idx_local[arg][-samplenum:]
            new_unlabeled_set = unlabeled_idx_local[arg][:-samplenum]
            new_labeled_set = np.concatenate([labeled_idx_local,new_labeled_set])
            new_labeled_idx.append(new_labeled_set)
            new_unlabeled_idx.append(new_unlabeled_set)
        new_labeled_idx = np.array(new_labeled_idx)
        new_unlabeled_idx = np.array(new_unlabeled_idx)

        return new_labeled_idx,new_unlabeled_idx

