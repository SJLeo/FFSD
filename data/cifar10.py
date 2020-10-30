from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transfroms

class Data:

    def __init__(self, opt):

        pin_memory = True

        trainTransfroms = transfroms.Compose([
            transfroms.RandomCrop(32, padding=4),
            transfroms.RandomHorizontalFlip(),
            transfroms.ToTensor(),
            transfroms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainDataset = CIFAR10(root=opt.dataroot, train=True, download=True, transform=trainTransfroms)
        self.trainLoader = DataLoader(
            dataset=trainDataset,
            batch_size=opt.train_batch_size,
            shuffle=opt.shuffle,
            num_workers=opt.num_threads,
            pin_memory=pin_memory,
        )

        testTransfroms = transfroms.Compose([
            transfroms.ToTensor(),
            transfroms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testDataset = CIFAR10(root=opt.dataroot, train=False, download=False, transform=testTransfroms)
        self.testLoader = DataLoader(
            dataset=testDataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=pin_memory,
        )

