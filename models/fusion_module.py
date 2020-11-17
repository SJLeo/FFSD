import torch.nn as nn

import math

class FusionModule(nn.Module):
    def __init__(self, channel, numclass, sptial, model_num=2):
        super(FusionModule, self).__init__()
        self.total_feature_maps = {}

        self.conv1 = nn.Conv2d(channel * model_num, channel * model_num, kernel_size=3, stride=1, padding=1, groups=channel*model_num, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * model_num)
        self.relu1 = nn.ReLU(True)
        self.conv1_1 = nn.Conv2d(channel * model_num, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)
        self.relu1_1 = nn.ReLU(True)
        self.avgpool2d = nn.AvgPool2d(sptial)

        self.fc2 = nn.Linear(channel, numclass)


        self.sptial = sptial

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.register_hook()

    def forward(self, x):

        x = self.relu1(self.bn1((self.conv1(x))))
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))

        x = self.avgpool2d(x)
        x = x.view(x.size(0), -1)
        out = self.fc2(x)

        return out

    def register_hook(self):

        self.extract_layers = ['relu1_1']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name+str(output.device)] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)


class TransFusionModule(nn.Module):

    def __init__(self, channel, sptial, model_num=2):
        super(TransFusionModule, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(True)
        self.conv1_1 = nn.Conv2d(channel, channel * model_num, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel * model_num)
        self.relu1_1 = nn.ReLU(True)

        self.sptial = sptial

    def forward(self, input):

        x = self.relu1(self.bn1((self.conv1(input))))
        out = self.relu1_1(self.bn1_1(self.conv1_1(x)))

        return out