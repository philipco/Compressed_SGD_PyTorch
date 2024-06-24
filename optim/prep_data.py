import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch
import random
from scipy.sparse import diags
from numpy import linalg
from scipy.stats import ortho_group
import math


def create_loaders(dataset_name, n_workers, batch_size, seed=42, val_ratio=0.1, common_ratio=0.0):

    train_data, test_data = load_data(dataset_name, n_workers)

    train_loader_workers = dict()
    n = len(train_data)

    # preparing iterators for workers and validation set
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    VAL_RATIO = val_ratio
    n_val = np.int(np.floor(VAL_RATIO * n))
    val_data = Subset(train_data, indices=indices[:n_val])
    
    COMMON_RATIO = common_ratio
    n_common = np.int(np.floor((COMMON_RATIO * (1 - VAL_RATIO) + VAL_RATIO) * n))
    indices_common = indices[n_val:n_common]
    indices = indices[n_common:]
    n = len(indices)
    a = np.int(np.floor(n / n_workers)) 
    top_ind = a * n_workers
    
    seq = range(a, top_ind, a)
    split = np.split(indices[:top_ind], seq)
    

    b = 0
    for ind in split:
        train_loader_workers[b] = DataLoader(Subset(train_data, np.concatenate((indices_common, ind))), batch_size=batch_size, shuffle=True)
        b = b + 1

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader_workers, val_loader, test_loader


def load_data(dataset_name, n_workers=None):

    if dataset_name == 'mnist':

        transform = transforms.ToTensor()

        train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)

        test_data = datasets.MNIST(root='data', train=False,
                                   download=True, transform=transform)
    elif dataset_name == 'cifar10':

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_data = datasets.CIFAR10(root='data', train=True,
                                      download=True, transform=transform)

        test_data = datasets.CIFAR10(root='data', train=False,
                                     download=True, transform=transform)
    elif dataset_name == 'cifar100':
        transform = transforms.ToTensor()  # add extra transforms
        train_data = datasets.CIFAR100(root='data', train=True,
                                       download=True, transform=transform)

        test_data = datasets.CIFAR100(root='data', train=False,
                                      download=True, transform=transform)
        
    elif dataset_name[:9] == 'quadratic':
        strs = dataset_name.split('_')
        d = int(strs[1])
        noise_scale = float(strs[2])
        regularizer = float(strs[3])
        train_data = zip(gen_similar_list_article(n_workers, d, noise_scale, regularizer), [torch.tensor(0) for _ in range(n_workers)])
        # train_data = zip(gen_random_list(n_workers, d), [torch.tensor(0) for _ in range(n_workers)])
        train_data = list(train_data)
        test_data = torch.zeros(1, d, d+1)
    else:
        raise ValueError(dataset_name + ' is not known.')

    return train_data, test_data

def gen_similar_list_article(nodes_count, d, noise_scale, regularizer):
    ksi_s, ksi_b = torch.from_numpy(np.random.normal(0, 1, size=nodes_count)), torch.from_numpy(np.random.normal(0, 1, size=nodes_count))
    nu_s, nu_b = torch.ones(nodes_count) + noise_scale*ksi_s, noise_scale*ksi_b

    b_list = []
    for i in range(nodes_count):
        mult = torch.zeros(d)
        mult[0] = -1 + nu_b[i]
        b_i = nu_s[i]/4*mult
        b_list.append(b_i)

    #tridiagonal matrix
    k = [torch.ones(d-1),-2*torch.ones(d),torch.ones(d-1)]
    offset = [-1,0,1]
    A_tri = -1*(torch.from_numpy(diags(k,offset).toarray()))

    A_list_similar = []
    A = torch.zeros((d, d))
    for i in range(nodes_count):
        A_list_similar.append(nu_s[i] / 4 * A_tri)
        A += A_list_similar[i]

    A = A/nodes_count
    lambda_min = torch.min(torch.abs(torch.linalg.eigvals(A)))

    for i in range(nodes_count):
        A_list_similar[i] += (regularizer - lambda_min)*torch.eye(d)

    x_0 = torch.zeros(d)
    x_0[0] = math.sqrt(d)

    result_list = [torch.cat((A_list_similar[i], b_list[i].unsqueeze(-1)), dim=-1) for i in range(nodes_count)]
    return result_list

def gen_random_list(nodes_count, d):
    A_list = []
    b_list = []
    mu = 1e-2
    L = 100
    for i in range(nodes_count):
        U = torch.from_numpy(ortho_group.rvs(dim=d))
        U = U.float()
        A = mu * torch.eye(d, dtype=torch.float32)
        A[0][0] = L
        A = torch.matmul(torch.matmul(U.T, A), U)
        A_list.append(A)
        b_list.append(torch.from_numpy(np.random.normal(0, 1, size=d)).float())
    result_list = [torch.cat((A_list[i], b_list[i].unsqueeze(-1)), dim=-1) for i in range(nodes_count)]
    return result_list