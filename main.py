import torch
from torch import nn
import numpy as np

from optim.train import tune_step_size, run_tuned_exp
from optim.models import MNISTNet, MNISTLogReg, resnet18, vgg11, SimplestNetwork, MNIST_CNN

from optim.utils import save_exp, load_exp, read_all_runs, create_exp
from utils.plotting import plot

from quant.quant import c_nat, random_dithering_wrap, rand_spars_wrap, \
    top_k_wrap, grad_spars_wrap, biased_unbiased_wrap, combine_two_wrap

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    with open("log.txt", 'a') as f:
        print("==== NEW RUN ====", file=f)

    dataset = 'mnist' # datasets, current options: 'mnist', 'cifar10', 'cifar100'
    model = 'mnist' # for saving purposes
    net = MNIST_CNN  # for the list of all models, see optim/models.py
    criterion = nn.CrossEntropyLoss()  # loss, which is considered
    epochs = 3  # number of epochs
    n_workers = 10  # number of workers
    batch_size = 64  # local batch size on each worker
    seed = 42  # fixed seed, which allows experiment replication
    lrs = np.array([0.1, 0.05, 0.01])  # learning rates, which are considered during tuning stage
    momentum = 0.9  # momentum for optimizer, default 0
    weight_decay = 0  # weight_decay for optimizer, default 0

    exp_name = dataset  # experiment name

    ### BiQSGD ###
    compression = {'wrapper': True, 'compression': random_dithering_wrap, 'p': np.inf, 's': 1, 'natural': False}
    master_compression = {'wrapper': True, 'compression': random_dithering_wrap, 'p': np.inf, 's': 1, 'natural': False}
    error_feedback = False

    name = exp_name + '_biqsqd'  # name based on which your experiment will be stored
    exp = create_exp(name, dataset, net, n_workers, epochs, seed, batch_size, lrs,
                     compression, error_feedback, criterion, master_compression, momentum, up_error_feedback=False,
                     down_error_feedback=False, use_up_memory=False, up_compression=True, down_compression=True)

    exp['lr'] = 0.1

    with open("log.txt", 'a') as f:
        print("===: ", name, file=f)
        print('Best learning rate {:2.4f}:'.format(exp['lr']), file=f)

    save_exp(exp)

    with open("log.txt", 'a') as f:
        print("==== RUNNING ====", file=f)

    run_tuned_exp(exp)
    exp_double = load_exp(exp_name + '_biqsqd')
    ##########################

    ### DORE ###
    compression = {'wrapper': True, 'compression': random_dithering_wrap, 'p': np.inf, 's': 1, 'natural': False}
    master_compression = {'wrapper': True, 'compression': random_dithering_wrap, 'p': np.inf, 's': 1, 'natural': False}

    name = exp_name + '_dore'  # name based on which your experiment will be stored
    exp = create_exp(name, dataset, net, n_workers, epochs, seed, batch_size, lrs,
                     compression, error_feedback, criterion, master_compression, momentum, up_error_feedback=True,
                     down_error_feedback=True, use_up_memory=True, up_compression=True, down_compression=True)

    exp['lr'] = 0.1

    with open("log.txt", 'a') as f:
        print("===: ", name, file=f)
        print('Best learning rate {:2.4f}:'.format(exp['lr']), file=f)

    save_exp(exp)

    with open("log.txt", 'a') as f:
        print("==== RUNNING ====", file=f)

    run_tuned_exp(exp)
    exp_dore = load_exp(exp_name + '_dore')
    ##########################

    ### ARTEMIS ###
    compression = {'wrapper': True, 'compression': random_dithering_wrap, 'p': np.inf, 's': 1, 'natural': False}
    master_compression = {'wrapper': True, 'compression': random_dithering_wrap, 'p': np.inf, 's': 1, 'natural': False}

    name = exp_name + '_artemis'  # name based on which your experiment will be stored
    exp = create_exp(name, dataset, net, n_workers, epochs, seed, batch_size, lrs,
                     compression, error_feedback, criterion, master_compression, momentum, up_error_feedback=False,
                     down_error_feedback=False, use_up_memory=True, up_compression=True, down_compression=True)

    exp['lr'] = 0.1

    with open("log.txt", 'a') as f:
        print("===: ", name, file=f)
        print('Best learning rate {:2.4f}:'.format(exp['lr']), file=f)

    save_exp(exp)

    with open("log.txt", 'a') as f:
        print("==== RUNNING ====", file=f)

    run_tuned_exp(exp)
    exp_artemis = load_exp(exp_name + '_artemis')

    #########################

    ### QSGD ###
    master_compression = {'wrapper': False, 'compression': None}

    name = exp_name + '_simple'  # name based on which your experiment will be stored
    exp = create_exp(name, dataset, net, n_workers, epochs, seed, batch_size, lrs,
                     compression, error_feedback, criterion, master_compression, momentum, up_error_feedback=False,
                     down_error_feedback=False, use_up_memory=False, up_compression=True, down_compression=False)

    exp['lr'] = 0.1

    with open("log.txt", 'a') as f:
        print("===: ", name, file=f)
        print('Best learning rate {:2.4f}:'.format(exp['lr']), file=f)

    save_exp(exp)

    with open("log.txt", 'a') as f:
        print("==== RUNNING ====", file=f)

    run_tuned_exp(exp)
    exp_simple = load_exp(exp_name + '_simple')
    ##########################

    ### DIANA ###
    master_compression = {'wrapper': False, 'compression': None}

    name = exp_name + '_diana'  # name based on which your experiment will be stored
    exp = create_exp(name, dataset, net, n_workers, epochs, seed, batch_size, lrs,
                     compression, error_feedback, criterion, master_compression, momentum, up_error_feedback=False,
                     down_error_feedback=False, use_up_memory=True, up_compression=True, down_compression=False)

    exp['lr'] = 0.1

    with open("log.txt", 'a') as f:
        print("===: ", name, file=f)
        print('Best learning rate {:2.4f}:'.format(exp['lr']), file=f)

    save_exp(exp)

    with open("log.txt", 'a') as f:
        print("==== RUNNING ====", file=f)

    run_tuned_exp(exp)
    exp_diana = load_exp(exp_name + '_diana')
    ##########################

    ### NO COMPRESSION ###
    compression = {'wrapper': False, 'compression': None}

    name = exp_name + '_sgd'  # name based on which your experiment will be stored
    exp = create_exp(name, dataset, net, n_workers, epochs, seed, batch_size, lrs,
                     compression, False, criterion, master_compression=compression, momentum=momentum, up_error_feedback=False,
                     down_error_feedback=False, use_up_memory=False, up_compression=False, down_compression=False)

    exp['lr'] = 0.1

    with open("log.txt", 'a') as f:
        print("===: ", name, file=f)
        print('Best learning rate {:2.4f}:'.format(exp['lr']), file=f)

    save_exp(exp)

    with open("log.txt", 'a') as f:
        print("==== RUNNING ====", file=f)

    run_tuned_exp(exp)
    exp_sgd = load_exp(exp_name + '_sgd')
    ##########################

    exp_type = 'test'

    # plot([exp_sgd], "train_loss", log_scale=False,
    #      legend=['SGD', 'BiQSGD', 'Artemis'],
    #      y_label='Test accuracy', file='train_loss.pdf')
    # plot([exp_sgd], "test_loss", log_scale=False,
    #      legend=['SGD', 'BiQSGD', 'Artemis'],
    #      y_label='Test accuracy', file='test_loss.pdf')
    # plot([exp_sgd], "test_acc", log_scale=False,
    #      legend=['SGD', 'BiQSGD', 'Artemis'],
    #      y_label='Test accuracy', file='test_acc.pdf')

    # plot([exp_sgd, exp_double, exp_artemis], "train_loss", log_scale=False,
    #      legend=['SGD', 'BiQSGD', 'Artemis'],
    #      y_label='Test accuracy', file='train_loss.pdf')
    # plot([exp_sgd, exp_double, exp_artemis], "test_loss", log_scale=False,
    #      legend=['SGD', 'BiQSGD', 'Artemis'],
    #      y_label='Test accuracy', file='test_loss.pdf')
    # plot([exp_sgd, exp_double, exp_artemis], "test_acc", log_scale=False,
    #      legend=['SGD', 'BiQSGD', 'Artemis'],
    #      y_label='Test accuracy', file='test_acc.pdf')

    plot([exp_sgd, exp_simple, exp_diana, exp_double, exp_artemis, exp_dore], "train_loss", log_scale=False,
         legend=['SGD', 'QSGD', 'Diana', 'BiQSGD', 'Artemis', 'Dore'],
         y_label='Test accuracy', file='train_loss.pdf')
    plot([exp_sgd, exp_simple, exp_diana, exp_double, exp_artemis, exp_dore], "test_loss", log_scale=False,
         legend=['SGD', 'QSGD', 'Diana', 'BiQSGD', 'Artemis', 'Dore'],
         y_label='Test accuracy', file='test_loss.pdf')
    plot([exp_sgd, exp_simple, exp_diana, exp_double, exp_artemis, exp_dore], "test_acc", log_scale=False,
         legend=['SGD', 'QSGD', 'Diana', 'BiQSGD', 'Artemis', 'Dore'],
         y_label='Test accuracy', file='test_acc.pdf')


