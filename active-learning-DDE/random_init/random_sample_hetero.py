import numpy as np 
import random
from sys import argv
import json

from data_preprocessing.cinic10.data_loader import  partition_data

CLIENTS = 10
NUM = 90000
INIT = 0.02
FINAL = 0.2
LOCAL = './cinic10/'
partition = 'hetero' #"hetero"
dataroot = '/data1/beichen_zhang/FedML/data/cinic10'
V = 0
rate = [0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20]
#print(rate)
rate.reverse()
#print(rate)



X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(0, dataroot, partition, CLIENTS, 0.5)

#print(type(net_dataidx_map))
#print(net_dataidx_map[0].shape)
#print(len(net_dataidx_map[0]))

final = np.array([np.array(net_dataidx_map[i]) for i in range(CLIENTS)])

fedsplit = final
now = fedsplit
for i in fedsplit:
	print(i.shape)
np.save(LOCAL + str(int(NUM)) + '_10_' + partition + '_v' + str(V)+ '.npy', final)






for i in range(len(rate)):
	newrate = rate[i]
	
	ll = []
	for j in range(CLIENTS):
		newnum = int(final[j].shape[0] * newrate)
		np.random.shuffle(now[j])
		nowidx = now[j]
		ll.append(nowidx[:newnum])
	fedsplit = np.array(ll)
	for i in fedsplit:
		print(i.shape)
	np.save(LOCAL + str(int(newrate * NUM / CLIENTS)) + '_' + str(CLIENTS) + '_' + partition + '_v' + str(V)+ '.npy', fedsplit)
	now = fedsplit

