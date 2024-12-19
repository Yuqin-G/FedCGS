import torch
import math
import numpy as np
import registry
import torch
import logging
from utils_fl import partition_data, DataLoader, DatasetSplit, get_dataset, eval, get_model
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

import argparse

if __name__ == "__main__":
    args = argparse.Namespace()
    logger = logger_config(log_path='./test.log', logging_name='test')
    device = torch.device('cuda:4')
    dataset_list = ["cifar100", "cifar10", "svhn"]
    beta_list = [0.05, 0.1, 0.5]
    for dataset in dataset_list:
        print(dataset)
        args.dataset = dataset
        for beta in beta_list:
            print(beta)
            num_classes, train_dataset, test_dataset = get_dataset(name=args.dataset, data_root='/home/guanzenghao/data/')
            print(len(train_dataset))
            train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
                train_dataset, test_dataset, 'dir', beta=beta, num_users=30, logger=logger, args=args)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                            shuffle=False, num_workers=4)                                 
            user_num = [len(user_groups[i]) for i in range(len(user_groups))]
            num_users = len(user_groups)

            train_loader_list = [DataLoader(DatasetSplit(train_dataset, user_groups[i]), batch_size=1024, shuffle=False) for i in range(num_users)]
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)

            model_name = "resnet18_imagenet"
            model, num_d = get_model(model_name, num_classes)
            model.eval()

            cs_l = []
            cc_l = []
            Ax_l = []
            Bx_l = []

            for i in range(len(train_loader_list)):
                cs1 = torch.zeros((num_classes, num_d)).double()
                cc1 = torch.zeros(num_classes)
                Ax1 = torch.zeros((num_d)).double()
                Bx1 = torch.zeros((num_d, num_d)).double()

                model = model.to(device)
                for i, (X, y) in enumerate(train_loader_list[i]):
                    with torch.no_grad():
                        X = X.to(device)
                        _, feature_t = model(X, return_features=True)
                        feature_t = feature_t.double()
                        feature_t = feature_t.cpu()
                        cs1.scatter_add_(0, y.unsqueeze(1).expand_as(feature_t).to(torch.int64), feature_t)
                        cc1.scatter_add_(0, y.to(torch.int64), torch.ones(len(y)))
                        Ax1 += feature_t.double().sum(axis=0)
                        Bx1 += feature_t.T @ feature_t
                    del feature_t, y, X
                cs_l.append(cs1)
                cc_l.append(cc1)
                Ax_l.append(Ax1)
                Bx_l.append(Bx1)

            cc = sum(cc_l)
            cs = sum(cs_l)
            Ax = sum(Ax_l)
            Bx = sum(Bx_l)

            N = cc.sum()
            mu = Ax / N

            mu_A = torch.ger(mu.T, Ax.double())
            A_mu = torch.ger(Ax.double().T, mu)
            mu_mu = torch.ger(mu.T, mu.double()) * N

            Sx = (Bx - mu_A - A_mu + mu_mu) / (N - 1)

            class_means = torch.div(cs, torch.reshape(cc, (-1, 1))).double()
            A, B, Aj, mu, S, N, Nj = Ax, Bx, class_means, mu, Sx, N, cc

            gamma_list = [0]
            for gamma in gamma_list:
                S_hat = S + gamma * torch.eye(num_d) 
                S_inv = np.linalg.inv(S_hat.detach())
                S_inv = torch.tensor(S_inv)
                eval(model, Aj, S_inv, N, Nj, test_loader, device)



