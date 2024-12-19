import torch
import numpy as np
import registry
import torch
import logging
import torch.nn.functional as F  #
from tqdm import tqdm
from numpy.typing import NDArray
from typing import List
from utils_fl import partition_data, DataLoader, DatasetSplit, get_dataset, get_model, test
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture as GMM

@dataclass
class GMMParameters:
    """GMM parameters."""
    label: NDArray
    means: NDArray
    weights: NDArray
    covariances: NDArray
    num_samples: NDArray

def learn_gmm(features_np, labels_np):
    g_list = []
    for label in np.unique(labels_np):
        cond_features = features_np[label == labels_np]
        if (
            len(cond_features) > n_mixtures
        ): 
            gmm = GMM(
                n_components=n_mixtures,
                covariance_type=cov_type,
                tol=tol,
                max_iter=max_iter,
            )
            gmm.fit(cond_features)
            g_list.append(
                GMMParameters(
                    label=np.array(label),
                    means=gmm.means_.astype("float16"),
                    weights=gmm.weights_.astype("float16"),
                    covariances=gmm.covariances_.astype("float16"),
                    num_samples=np.array(len(cond_features)),
                )
            )
    return g_list

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
    device = torch.device('cuda:6')
    dataset_list = ["svhn", "cifar10", "cifar100"]

    n_mixtures = 10
    cov_type = "diag"
    tol = 1e-12
    max_iter = 10000
    beta_list = [0.05]

    for dataset in dataset_list:
        print(dataset)
        args.dataset = dataset
        for beta in beta_list:
            print("beta", beta)
            num_classes, train_dataset, test_dataset = get_dataset(name=args.dataset, data_root='/home/guanzenghao/data/')
            train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
                train_dataset, test_dataset, 'dir', beta=beta, num_users=10, logger=logger, args=args)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                                            shuffle=False, num_workers=4, pin_memory=True)                                 
            user_num = [len(user_groups[i]) for i in range(len(user_groups))]
            num_users = len(user_groups)
            train_loader_list = [DataLoader(DatasetSplit(train_dataset, user_groups[i]), batch_size=1024, shuffle=False, num_workers=4) for i in range(num_users)]
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)

            model_name = "resnet18_imagenet"
            model, num_d = get_model(model_name, num_classes)
            model.eval()
            model = model.to(device)

            features_local = []
            labels_local = []
            for train_loader in train_loader_list:
                features, labels = [], []
                for batch_samples, batch_label in train_loader:
                    with torch.no_grad():
                        _, feature = model(batch_samples.to(device), return_features=True)
                    features.append(feature.cpu().detach().numpy())
                    labels.append(batch_label.cpu().detach().numpy())

                features_np = np.concatenate(features, axis=0).astype("float64")
                labels_np = np.concatenate(labels)
                features_local.append(features_np)
                labels_local.append(labels_np)

            gmm_list = []
            for f, l in zip(features_local, labels_local):
                gmm_list.append(learn_gmm(f, l))

            synthetic_features_dataset = []
            for gmm_ in gmm_list:
                for gmm_parameter in gmm_:
                    gmm = GMM(
                        n_components=n_mixtures,
                        covariance_type=cov_type,
                        tol=tol,
                        max_iter=max_iter,
                    )
                    gmm.means_ = gmm_parameter.means.astype("float32")
                    gmm.weights_ = gmm_parameter.weights.astype("float32")
                    gmm.covariances_ = gmm_parameter.covariances.astype("float32")

                    syn_features, _ = gmm.sample(gmm_parameter.num_samples)
                    syn_features = torch.tensor(syn_features, dtype=torch.float32)
                    gmm_labels = torch.tensor(
                        [int(gmm_parameter.label)] * int(gmm_parameter.num_samples)
                    )

                    synthetic_features_dataset += list(zip(syn_features, gmm_labels))

            synthetic_features_dataset = [
                {"img": img, "label": label} for img, label in synthetic_features_dataset
            ]
            synthetic_loader = DataLoader(
                synthetic_features_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            model_name = "resnet18_imagenet"
            model = registry.get_model(model_name, num_classes=num_classes)
            model.train()
            model = model.to(device)

            optimizer_ft_net = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9) 
            for epoch in range(50):
                trainloader_ft = synthetic_loader
                for data_batch in trainloader_ft:
                    images, labels = data_batch["img"], data_batch["label"]
                    images, labels = images.to(device), labels.to(device)
                    
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    for param in model.fc.parameters():
                        param.requires_grad = True

                    outputs = model.fc(images)
                    loss_net = F.cross_entropy(outputs, labels)
                    
                    optimizer_ft_net.zero_grad()
                    loss_net.backward()
                    
                    optimizer_ft_net.step()

            test(model, test_loader, device)
            print()