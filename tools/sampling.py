import matplotlib.pyplot as plt

import torchvision
import numpy as np
import os
import torch

train_data = torchvision.datasets.FashionMNIST(root='../fashionmnist', train=True, download=True)

print(train_data[0])

num_samples = len(train_data)
part_size = num_samples // 2
indices = list(range(num_samples))
np.random.shuffle(indices)

part1_indices = indices[:part_size]
part2_indices = indices[part_size:]

os.makedirs("../data", exist_ok=True)

part1_data = [train_data[i] for i in part1_indices]
part2_data = [train_data[i] for i in part2_indices]

torch.save(part1_data, "../data/part1.pth")
torch.save(part2_data, "../data/part2.pth")

load1 = torch.load('../data/part1.pth')
print(load1[0])

