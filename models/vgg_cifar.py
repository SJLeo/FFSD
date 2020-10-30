import torch
from torchsummary import summary
import torch.nn as nn
from models.fusion_module import TransFusionModule

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, num_classes=100, leader=False, trans_fusion_info=None):
        super(VGG, self).__init__()

        self.total_feature_maps = {}
        self.leader = leader

        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, num_classes)

        if self.leader:
            if trans_fusion_info is not None:
                self.trans_fusion_module = TransFusionModule(trans_fusion_info[0], 2, model_num=trans_fusion_info[1])
            else:
                self.trans_fusion_module = TransFusionModule(512, 2)

        self.reset_parameters()
        self.register_hook()

    def forward(self, x):

        x = self.features(x)
        if self.leader:
            trans_fusion_output = self.trans_fusion_module(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.leader:
            return x, trans_fusion_output
        else:
            return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def register_hook(self):

        self.extract_layers = ['features.5', 'features.12', 'features.22', 'features.33', 'features.42']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg(**kwargs):

    return VGG(**kwargs)

if __name__ == '__main__':

    model = VGG()
    summary(model, (3, 32, 32), batch_size=1)
    print(model)

    input = torch.randn((1, 3, 32, 32))
    output = model(input)
    print(output.size())

