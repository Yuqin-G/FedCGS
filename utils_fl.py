import os
import pdb
import wandb
import torch
from PIL import Image
import os, random, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import copy
import torch
import numpy as np
from torchvision import datasets, transforms
import registry

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

def record_net_data_stats(y_train, net_dataidx_map, logger=None):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        logger.info("Client ID %3d: %s" %(net_i, tmp))

    # logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def load_data(trn_dst, tst_dst, args=None):
    if args.dataset == "tiny":
        # len(trn_dst)
        train_load = torch.utils.data.DataLoader(trn_dst, batch_size=len(trn_dst), shuffle=False, num_workers=0)
        test_load = torch.utils.data.DataLoader(tst_dst, batch_size=len(tst_dst), shuffle=False, num_workers=0)
        train_itr = train_load.__iter__();
        test_itr = test_load.__iter__()
        # labels are of shape (n_data,)
        X_train, y_train = train_itr.__next__()
        X_test, y_test = test_itr.__next__()
        X_train = X_train.numpy();
        y_train = y_train.numpy().reshape(-1, 1)
        X_test = X_test.numpy();
        y_test = y_test.numpy().reshape(-1, 1)
        # pdb.set_trace()
    elif args.dataset == "svhn":
        X_train, y_train = trn_dst.data, trn_dst.labels
        X_test, y_test = tst_dst.data, tst_dst.labels
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        X_train, y_train = trn_dst.data, trn_dst.targets
        X_test, y_test = tst_dst.data, tst_dst.targets
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test, trn_dst, tst_dst

def partition_data(trn_dst,tst_dst, partition, beta=0.4, num_users=5,logger=None,args=None):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(trn_dst, tst_dst, args)
    data_size = y_train.shape[0]
    # pdb.set_trace()
    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dir":
        min_size = 0
        min_require_size = 10
        K = 100
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "n_cls":
        n_client = n_parties
        n_cls = np.unique(y_test).shape[0]
        alpha = beta

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))
        if n_client <= 5:
            for i in range(n_client):
                for j in range(int(alpha)):
                    cls_priors[i][int((alpha*i+j))%n_cls] = 1.0 / alpha
        else:
            for i in range(n_client):
                cls_priors[i][random.sample(range(n_cls), int(alpha))] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []
        # pdb.set_trace()
        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    continue
                else:
                    cls_amount[cls_label] -= 1
                    net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                    break

    
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logger)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)
        # return torch.tensor(image), torch.tensor(label)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.names = names
        self.labels = labels
        self._img_transformer = img_transformer

    def get_image(self, index):
        name = self.names[index]
        img = Image.open(name).convert('RGB')
        return self._img_transformer(img)

    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index]), self.names[index]

    def __len__(self):
        return len(self.names)

def get_dataset(name: str, data_root: str='data', return_transform=False, split=['A', 'B', 'C', 'D']):
    import os
    from torchvision import datasets, transforms as T
    from registry import DatasetFromDir, NORMALIZE_DICT
    name = name.lower()
    data_root = os.path.expanduser( data_root )
    data_path = os.path.join(data_root, name)
    if name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_root = os.path.join(data_root)
        train_dst = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_path, train=False, download=True, transform=val_transform)
    elif name == "cifar100":
        num_classes = 100
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_root = os.path.join( data_root)
        train_dst = datasets.CIFAR100(data_path, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_path, train=False, download=True, transform=val_transform)
    elif name=='svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'svhn' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst

import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models import efficientnet_b0

def get_model(model_name, num_classes):
    num_d = 0
    if (model_name == "resnet18_imagenet"):
        model = registry.get_model(model_name, num_classes=num_classes)
        num_d = 512
    return model, num_d

def test(model, test_loader, device="cuda:0"):
    from tqdm import tqdm
    model.eval()
    model.to(device)
    correct = 0
    g_acc = 0
    with torch.no_grad(): 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # output = model.fc((t))
            output = model(data)
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        torch.cuda.empty_cache()
        g_acc = 100. * correct / len(test_loader.dataset)

        print(g_acc)
    return g_acc


def eval(model, Aj, S_inv, N, Nj, test_loader, device="cuda:0"):
    W = Aj @ torch.tensor(S_inv)
    b = torch.zeros(Aj.size(0))
    for i in range(Aj.size(0)):
        b[i] = -(Aj[i].T @ S_inv @ Aj[i]) / 2 + torch.log(Nj[i] / N)
    import copy
    model1 = copy.deepcopy(model)
    model1.fc.weight.data = W.float().to(device)
    model1.fc.bias.data = b.float().to(device)
    
    g_acc = test(model1, test_loader)
    return g_acc

