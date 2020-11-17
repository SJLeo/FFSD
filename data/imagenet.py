import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class Data:
    def __init__(self, opt):
        pin_memory = True

        scale_size = 224

        traindir = os.path.join(opt.data_path, 'train')
        valdir = os.path.join(opt.data_path, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.trainLoader = DataLoader(
            trainset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testLoader = DataLoader(
            testset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True)