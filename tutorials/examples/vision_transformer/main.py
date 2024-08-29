import argparse
import os

from solver import Solver
from torch.backends import cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torchvision

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Data loader
    train_dir = os.path.join(config.data_path,'train')
    test_dir = os.path.join(config.data_path,'val')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Set data loader
    trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True,
                                        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                          shuffle=True, num_workers=config.workers)
    testset = torchvision.datasets.CIFAR10(root=test_dir, train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                        shuffle=False, num_workers=config.workers)

    # Solver for training and testing ViT
    solver = Solver(trainloader, testloader, config)

    # Add to train / test mode
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vision transformer')

    parser.add_argument('--data', type=str, default='cifar10', metavar='N',
                        help='data')
    parser.add_argument('--data_path', type=str, default='./path', metavar='N',
                        help='input data path') 
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--workers', type=int, default=1, metavar='N',
                        help='num_workers') 

    config = parser.parse_args()
    cudnn.benchmark=True

    main(config)