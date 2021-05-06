import torch
import random
import numpy as np
import os
from collections import Counter


from torchvision.models import resnet18, resnet50, resnet101, resnet152
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from torchvision.transforms import CenterCrop


def set_seed(args, use_gpu, print_out=True):
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed(args.seed)


def count_correct_top_k(scores, labels, k):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    top_k_scores = torch.argsort(scores, axis=-1, descending=True)[:, :k]
    labels = labels.view(len(labels), 1)
    return torch.eq(labels, top_k_scores).sum()


def count_correct_average_k(probas, labels, lmbda):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    gt_probas = torch.gather(probas, dim=1, index=labels.unsqueeze(-1))
    res = torch.sum((gt_probas) >= lmbda)
    return res


def compute_lambda_batch(batch_proba, k):
    sorted_probas, _ = torch.sort(torch.flatten(batch_proba), descending=True)
    batch_lambda = 0.5 * (sorted_probas[len(batch_proba) * k - 1] + sorted_probas[len(batch_proba) * k])
    return batch_lambda


def load_model(model, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    model.load_state_dict(d['model'])
    return d['epoch']


def load_optimizer(optimizer, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    optimizer.load_state_dict(d['optimizer'])


def save(model, optimizer, epoch, location):
    dir = os.path.dirname(location)
    if not os.path.exists(dir):
        os.makedirs(dir)

    d = {'epoch': epoch,
         'model': model.state_dict(),
         'optimizer': optimizer.state_dict()}
    torch.save(d, location)


def decay_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    print('Switching lr to {}'.format(optimizer.param_groups[0]['lr']))
    return optimizer


def update_optimizer(optimizer, lr_schedule, epoch):
    if epoch in lr_schedule:
        optimizer = decay_lr(optimizer)
    return optimizer


def get_model(args, n_classes):
    model_dict = {'resnet18': resnet18, 'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152}

    model = model_dict[args.model](pretrained=False)
    # Change last layer to account for number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)

    return model


class Plantnet(ImageFolder):
    def __init__(self, root, split, **kwargs):
        self.root = root
        self.split = split
        super().__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)


class MaxCenterCrop:
    def __call__(self, sample):
        min_size = min(sample.size[0], sample.size[1])
        return CenterCrop(min_size)(sample)


def get_data(args):

    transform = transforms.Compose(
        [MaxCenterCrop(), transforms.Resize(args.size_image), transforms.ToTensor()])

    trainset = Plantnet(args.root, 'train', transform=transform)
    train_class_to_num_instances = Counter(trainset.targets)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

    valset = Plantnet(args.root, 'val', transform=transform)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)

    testset = Plantnet(args.root, 'test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    val_class_to_num_instances = Counter(valset.targets)
    n_classes = len(trainset.classes)

    dataset_attributes = {'n_train': len(trainset), 'n_val': len(valset), 'n_test': len(testset), 'n_classes': n_classes,
                          'lr_schedule': [],
                          'class2num_instances': {'train': train_class_to_num_instances,
                                                  'val': val_class_to_num_instances},
                          'class_to_idx': trainset.class_to_idx}

    return trainloader, valloader, testloader, dataset_attributes