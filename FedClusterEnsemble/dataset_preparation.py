"""Download data and partition data with different partitioning strategies."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST,SVHN
import pickle


# Custom Dataset class for MotionSense
class MotionSenseDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

def load_MotionSense_dataset(transform_train=None, transform_test=None):
    x_train_list, y_train_list = [], []
    x_test_list = []
    y_test_list = []

    for cid in range(24):  # Assuming 24 clients
        # Load train data
        with open(f'data/motion_sense/{cid+1}_train.pickle', 'rb') as train_file:
            train = pickle.load(train_file)
        y_train = train['activity'].values
        train.drop(['activity', 'subject', 'trial'], axis=1, inplace=True)
        x_train = train.values
        
        # Load test data
        with open(f'data/motion_sense/{cid+1}_test.pickle', 'rb') as test_file:
            test = pickle.load(test_file)
        y_test = test['activity'].values
        test.drop(['activity', 'subject', 'trial'], axis=1, inplace=True)
        x_test = test.values
        
        # Append the data to the lists
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)

    # Concatenate all client data
    x_train_full = np.concatenate(x_train_list)
    y_train_full = np.concatenate(y_train_list)
    x_test_full = np.concatenate(x_test_list)
    y_test_full = np.concatenate(y_test_list)

    # Create trainset and testset using a PyTorch Dataset-like structure
    trainset = MotionSenseDataset(x_train_full, y_train_full)
    testset = MotionSenseDataset(x_test_full, y_test_full)

    return trainset, testset

def _download_data(dataset_name="emnist") -> Tuple[Dataset, Dataset]:
    """Download the requested dataset. Currently supports cifar10, mnist, and fmnist.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    trainset, testset = None, None
    if dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "fmnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "motion_sense":
        
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset, testset = load_MotionSense_dataset(transform_train,transform_test)
    elif dataset_name == "svhn":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
            ]
        )

        trainset = SVHN(
            root="data/svhn", split="train", download=True, transform=transform
        )
        testset = SVHN(
            root="data/svhn", split="test", download=True, transform=transform
        )
    else:
        raise NotImplementedError

    return trainset, testset


# pylint: disable=too-many-locals
def partition_data(
    num_clients, similarity=1.0, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition the dataset into subsets for each client.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    similarity: float
        Parameter to sample similar data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    s_fraction = int(similarity * len(trainset))
    prng = np.random.default_rng(seed)
    idxs = prng.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))

    # sample iid data per client from iid_trainset
    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    if similarity == 1.0:
        return trainsets_per_client, testset

    tmp_t = rem_trainset.dataset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    targets = tmp_t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: List[List] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i % num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    rem_trainsets_per_client: List[List] = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(
                    Subset(rem_trainset.dataset, act_idx)
                )
                ids += 1

    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset(
            [trainsets_per_client[i]] + rem_trainsets_per_client[i]
        )

    return trainsets_per_client, testset


def partition_data_dirichlet(
    num_clients, alpha, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset


def partition_data_label_quantity(
    num_clients, labels_per_client, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition the data according to the number of labels per client.

    Logic from https://github.com/Xtra-Computing/NIID-Bench/.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    num_labels_per_client: int
        Number of labels per client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    prng = np.random.default_rng(seed)

    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients: List[List] = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset

def partition_data_by_label_groups(
    num_clients, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition the data so that clients are divided into groups with specific labels.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    seed : int, optional
        Used to set a fixed seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    # Download the dataset
    trainset, testset = _download_data(dataset_name)
    prng = np.random.default_rng(seed)

    # Get the targets/labels correctly depending on the dataset
    if hasattr(trainset, 'targets'):
        targets = trainset.targets  # For CIFAR-10, MNIST, etc.
    elif hasattr(trainset, 'labels'):
        targets = trainset.labels  # For SVHN
    else:
        raise ValueError(f"Dataset {dataset_name} does not contain targets or labels.")

    # Convert targets to numpy array if needed
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    # Determine the number of classes
    num_classes = len(set(targets))

    # Define label groups
    label_groups = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8, 9]
    }

    # Assign clients to label groups
    clients_per_group = num_clients // 3
    client_groups = {i: list(range(i * clients_per_group, (i + 1) * clients_per_group)) for i in range(3)}

    # Adjust for any remaining clients
    remaining_clients = num_clients % 3
    for i in range(remaining_clients):
        client_groups[i].append(3 * clients_per_group + i)

    # Initialize lists for client data indices
    idx_clients: List[List[int]] = [[] for _ in range(num_clients)]

    # Partition data by label groups
    for group, labels in label_groups.items():
        for label in labels:
            idx_k = np.where(targets == label)[0]
            prng.shuffle(idx_k)
            idx_k_split = np.array_split(idx_k, len(client_groups[group]))
            for idx, client in enumerate(client_groups[group]):
                idx_clients[client].extend(idx_k_split[idx].tolist())

    # Create Subsets for each client
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]

    return trainsets_per_client, testset

if __name__ == "__main__":
    partition_data(100, 0.1)
