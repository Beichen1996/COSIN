import numpy as np
import torch


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def aggregation(
        X,
        num_clusters,
        distance='cosine',
        tol=1e-4,
        device = None
):
    """
    perform 
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    #X = X.float()

    # transfer to device
    #X = X.to(device)

    # initialize
    dis_min = float('inf')
    initial_state_best = initialize(X, num_clusters).to(device)
    for i in range(20):
        initial_state = initialize(X, num_clusters).to(device)
        dis = pairwise_distance_function(X, initial_state).sum()
        #print('test',initial_state)
        if dis < dis_min:
            #print(dis)
            dis_min = dis
            initial_state_best = initial_state
        #else:
        #    print('no', dis_min, dis)

    initial_state = initial_state_best
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)
        #print('test',initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if iteration > 300:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster, initial_state


def pairwise_distance(data1, data2):
    # transfer to device

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2):
    # transfer to device

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / (A.norm(dim=-1, keepdim=True) + 0.0001)
    B_normalized = B / (B.norm(dim=-1, keepdim=True) + 0.0001)

    cosine = A_normalized * B_normalized
    #print(A_normalized.shape, B_normalized.shape, cosine.shape)

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    cosine_dis = (cosine_dis  + 1)/2
    #print(cosine_dis.shape)
    #print(cosine_dis)
    cosine_dis[cosine_dis < 0.2] = 0 #edge weight compute
    #print(cosine_dis)
    return cosine_dis
