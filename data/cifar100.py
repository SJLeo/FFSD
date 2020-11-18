from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

class Data:

    def __init__(self, opt):

        pin_memory = True

        trainTransforms = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        testTransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])


        trainDataset = CIFAR100(root=opt.dataroot, train=True, download=True, transform=trainTransforms)
        self.trainLoader = DataLoader(
            dataset=trainDataset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=pin_memory,
        )

        testDataset = CIFAR100(root=opt.dataroot, train=False, download=False, transform=testTransforms)
        self.testLoader = DataLoader(
            dataset=testDataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=pin_memory,
        )

