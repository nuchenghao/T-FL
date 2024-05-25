import torch
from torchvision import transforms
import torchvision
from torch.utils import data


def load_data_fashion_mnist(batchSize, option, name="server", resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    if option == 'train':
        mnist = torch.load(name + '.pt')
    elif option == 'test':
        mnist = torchvision.datasets.FashionMNIST(
            root="./data", train=False, transform=trans, download=True)

    return data.DataLoader(mnist, batchSize, shuffle=True)
