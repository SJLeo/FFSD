import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class Data:

    def __init__(self, args):

        traindir = os.path.join(args.dataroot, 'ILSVRC2012_img_train')
        valdir = os.path.join(args.dataroot, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        self.trainLoader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=(self.train_sampler is None),
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   sampler=self.train_sampler)

        self.testLoader = torch.utils.data.DataLoader(datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )