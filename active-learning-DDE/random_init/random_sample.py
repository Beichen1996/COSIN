import numpy as np 
import random
from sys import argv
import json

from data_preprocessing.cinic10.data_loader import  partition_data

CLIENTS = 10 #number of clinets
NUM = 50000 # number of samples in the data pool 50000 for cifar
INIT = 0.02 #sampling rate at initilization
FINAL = 0.2 #label budget
LOCAL = './cifar_10/' # cifar_100 or cifar_10 or whatever
partition = 'homo' #"hetero"
dataroot = '/data1/beichen_zhang/FedML/data/cinic10' #cinic10, cifar10, cifar100
V = 0 #version index
rate = [0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20]
#print(rate)
rate.reverse()
#print(rate)



X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(0, dataroot, partition, CLIENTS, 0.5)

#print(type(net_dataidx_map))
#print(net_dataidx_map[0].shape)
#print(len(net_dataidx_map[0]))

ll = [net_dataidx_map[i] for i in range(CLIENTS)]
fedsplit = np.stack(ll)
now = fedsplit
print(fedsplit.shape)
np.save(LOCAL + '50000_10_' + partition + '_v' + str(V)+ '.npy', fedsplit)

for i in range(len(rate)):
	newrate = rate[i]
	newnum = int(NUM / CLIENTS * newrate)
	ll = []
	for j in range(CLIENTS):
		np.random.shuffle(now[j])
		nowidx = now[j]
		ll.append(nowidx[:newnum])
	fedsplit = np.stack(ll)
	print(fedsplit.shape)
	np.save(LOCAL + str(newnum) + '_10_homo_v' + str(V)+ '.npy', fedsplit)
	now = fedsplit


'''
num = int(argv[1])
sample = int(argv[2])
version = int(argv[3])

all_indices = set(np.arange(num))

initial_indices = random.sample(all_indices, sample)



c_list = np.array(initial_indices)
np.save(argv[2] + 'of' + argv[1] +'_v'+argv[3]+ '.npy', c_list )
'''