import torch

import time

import utils.util as utils
from data import create_dataLoader
from models import create_model

fusion_channel = {
    'resnet20': 64,
    'resnet32': 64,
    'resnet56': 64,
    'wideresnet1602': 128,
    'wideresnet4002': 128,
    'densenet': 456,
    'googlenet': 1024,
    'resnet18': 512,
    'resnet34': 512
}

class Tester():

    def __init__(self, opt):

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if torch.cuda.is_available() else 'cpu'
        self.model_num = self.opt.model_num

        self.fusion_channel = fusion_channel[self.opt.model]

        dataLoader = create_dataLoader(opt)
        self.testLoader = dataLoader.testLoader

        self.leader_model = create_model(self.opt, leader=True, trans_fusion_info=(self.fusion_channel, self.model_num)).to(self.device)
        ckpt = torch.load(self.opt.load_path, map_location=self.device)
        self.leader_model.load_state_dict(ckpt['weight'])

    def test(self):

        leader_accuracy = utils.AverageMeter()
        self.leader_model.eval()

        start_time = time.time()
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(self.testLoader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                leader_output, _ = self.leader_model(inputs)

                leader_prec = utils.accuracy(leader_output, labels.data, topk=(1, ))
                leader_accuracy.update(leader_prec[0], inputs.size(0))

            current_time = time.time()

            print('Model[{}]:\tAccuracy {:.2f}%\tTime {:.2f}s'
                  .format('Leader', float(leader_accuracy.avg), (current_time - start_time)))