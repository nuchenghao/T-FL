import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import numpy as np

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def dirichlet_split(dataset, num_subsets, alpha):
    # 获取所有标签
    labels = np.array([data[1] for data in dataset])
    unique_labels = np.unique(labels)

    # 为每个类别生成狄利克雷分布
    dirichlet_distributions = {label: np.random.dirichlet([alpha] * num_subsets) for label in unique_labels}

    # 根据狄利克雷分布为每张图片分配子数据集
    subsets_indices = {i: [] for i in range(num_subsets)}
    for idx, label in enumerate(labels):
        # 根据图片的类别选择子数据集
        subset_idx = np.argmax(np.random.multinomial(1, dirichlet_distributions[label]))
        subsets_indices[subset_idx].append(idx)

    # 创建子数据集
    subsets = [torch.utils.data.Subset(dataset, indices) for indices in subsets_indices.values()]
    return subsets


def iid_split(dataset, num_subsets):
    dataset_size = len(dataset)
    sub_dataset_size = dataset_size // num_subsets
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    subsets_indices = [indices[i:i + sub_dataset_size] for i in range(0, dataset_size, sub_dataset_size)]
    subsets = [torch.utils.data.Subset(dataset, indices) for indices in subsets_indices]
    return subsets


def split_data_fashion_mnist(batch_size, resize=None):

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)

    # dirichlet_split
    alpha = 0.7  # 可以调整alpha值来控制分布的集中或分散程度
    num_subsets = 5

    subsets = dirichlet_split(mnist_train, num_subsets, alpha)
    for i in range(num_subsets):
        torch.save(subsets[i], 'noniidclient{}.pt'.format(i))

    # iid split

    subsets = iid_split(mnist_train, num_subsets)
    for i in range(num_subsets):
        torch.save(subsets[i], 'iid{}.pt'.format(i))
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)

    return (data.DataLoader(subsets[0], batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))
