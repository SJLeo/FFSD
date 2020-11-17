import torch
import torch.nn as nn

from models.fusion_module import TransFusionModule

import math


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red1, n5x5red2, n5x5, pool_planes, tmp_name):
        super(Inception, self).__init__()
        self.tmp_name=tmp_name

        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            conv1x1 = nn.Conv2d(in_planes, n1x1, kernel_size=1)
            conv1x1.tmp_name = self.tmp_name

            self.branch1x1 = nn.Sequential(
                conv1x1,
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:
            conv3x3_1=nn.Conv2d(in_planes, n3x3red, kernel_size=1)
            conv3x3_2=nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1)
            conv3x3_1.tmp_name = self.tmp_name
            conv3x3_2.tmp_name = self.tmp_name

            self.branch3x3 = nn.Sequential(
                conv3x3_1,
                nn.BatchNorm2d(n3x3red),
                nn.ReLU(True),
                conv3x3_2,
                nn.BatchNorm2d(n3x3),
                nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:
            conv5x5_1 = nn.Conv2d(in_planes, n5x5red1, kernel_size=1)
            conv5x5_2 = nn.Conv2d(n5x5red1, n5x5red2, kernel_size=3, padding=1)
            conv5x5_3 = nn.Conv2d(n5x5red2, n5x5, kernel_size=3, padding=1)
            conv5x5_1.tmp_name = self.tmp_name
            conv5x5_2.tmp_name = self.tmp_name
            conv5x5_3.tmp_name = self.tmp_name

            self.branch5x5 = nn.Sequential(
                conv5x5_1,
                nn.BatchNorm2d(n5x5red1),
                nn.ReLU(True),
                conv5x5_2,
                nn.BatchNorm2d(n5x5red2),
                nn.ReLU(True),
                conv5x5_3,
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            conv_pool = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
            conv_pool.tmp_name = self.tmp_name

            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                conv_pool,
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

    def forward(self, x):
        out = []
        y1 = self.branch1x1(x)
        out.append(y1)

        y2 = self.branch3x3(x)
        out.append(y2)

        y3 = self.branch5x5(x)
        out.append(y3)

        y4 = self.branch_pool(x)
        out.append(y4)
        return torch.cat(out, 1)


class GoogLeNet(nn.Module):
    def __init__(self, block=Inception, filters=None, layer_cfg=None, num_classes=100, leader=False, trans_fusion_info=None):
        super(GoogLeNet, self).__init__()

        self.total_feature_maps = {}
        self.leader = leader
        self.layer_cfg = layer_cfg

        conv_pre = nn.Conv2d(3, 192, kernel_size=3, padding=1)
        conv_pre.tmp_name='pre_layer'
        self.pre_layers = nn.Sequential(
            conv_pre,
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        if filters is None:
            filters = [
                [64, 128, 32, 32],
                [128, 192, 96, 64],
                [192, 208, 48, 64],
                [160, 224, 64, 64],
                [128, 256, 64, 64],
                [112, 288, 64, 64],
                [256, 320, 128, 128],
                [256, 320, 128, 128],
                [384, 384, 128, 128]
            ]

        self.filters=filters

        self.inception_a3 = block(in_planes=192,
                                  n1x1=filters[0][0],
                                  n3x3red=96 if self.layer_cfg is None else self.layer_cfg[0],
                                  n3x3=filters[0][1],
                                  n5x5red1=16 if self.layer_cfg is None else self.layer_cfg[1],
                                  n5x5red2=filters[0][2] if self.layer_cfg is None else self.layer_cfg[2],
                                  n5x5=filters[0][2],
                                  pool_planes=filters[0][3],
                                  tmp_name='a3')
        self.inception_b3 = block(in_planes=sum(filters[0]),
                                  n1x1=filters[1][0],
                                  n3x3red=128 if self.layer_cfg is None else self.layer_cfg[3],
                                  n3x3=filters[1][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[4],
                                  n5x5red2=filters[1][2] if self.layer_cfg is None else self.layer_cfg[5],
                                  n5x5=filters[1][2],
                                  pool_planes=filters[1][3],
                                  tmp_name='b3')

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception_a4 = block(in_planes=sum(filters[1]),
                                  n1x1=filters[2][0],
                                  n3x3red=96 if self.layer_cfg is None else self.layer_cfg[6],
                                  n3x3=filters[2][1],
                                  n5x5red1=16 if self.layer_cfg is None else self.layer_cfg[7],
                                  n5x5red2=filters[2][2] if self.layer_cfg is None else self.layer_cfg[8],
                                  n5x5=filters[2][2],
                                  pool_planes=filters[2][3],
                                  tmp_name='a4')
        self.inception_b4 = block(in_planes=sum(filters[2]),
                                  n1x1=filters[3][0],
                                  n3x3red=112 if self.layer_cfg is None else self.layer_cfg[9],
                                  n3x3=filters[3][1],
                                  n5x5red1=24 if self.layer_cfg is None else self.layer_cfg[10],
                                  n5x5red2=filters[3][2] if self.layer_cfg is None else self.layer_cfg[11],
                                  n5x5=filters[3][2],
                                  pool_planes=filters[3][3],
                                  tmp_name='b4')
        self.inception_c4 = block(in_planes=sum(filters[3]),
                                  n1x1=filters[4][0],
                                  n3x3red=128 if self.layer_cfg is None else self.layer_cfg[12],
                                  n3x3=filters[4][1],
                                  n5x5red1=24 if self.layer_cfg is None else self.layer_cfg[13],
                                  n5x5red2=filters[4][2] if self.layer_cfg is None else self.layer_cfg[14],
                                  n5x5=filters[4][2],
                                  pool_planes=filters[4][3],
                                  tmp_name='c4')
        self.inception_d4 = block(in_planes=sum(filters[4]),
                                  n1x1=filters[5][0],
                                  n3x3red=144 if self.layer_cfg is None else self.layer_cfg[15],
                                  n3x3=filters[5][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[16],
                                  n5x5red2=filters[5][2] if self.layer_cfg is None else self.layer_cfg[17],
                                  n5x5=filters[5][2],
                                  pool_planes=filters[5][3],
                                  tmp_name='d4')
        self.inception_e4 = block(in_planes=sum(filters[5]),
                                  n1x1=filters[6][0],
                                  n3x3red=160 if self.layer_cfg is None else self.layer_cfg[18],
                                  n3x3=filters[6][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[19],
                                  n5x5red2=filters[6][2] if self.layer_cfg is None else self.layer_cfg[20],
                                  n5x5=filters[6][2],
                                  pool_planes=filters[6][3],
                                  tmp_name='e4')

        self.inception_a5 = block(in_planes=sum(filters[6]),
                                  n1x1=filters[7][0],
                                  n3x3red=160 if self.layer_cfg is None else self.layer_cfg[21],
                                  n3x3=filters[7][1],
                                  n5x5red1=32 if self.layer_cfg is None else self.layer_cfg[22],
                                  n5x5red2=filters[7][2] if self.layer_cfg is None else self.layer_cfg[23],
                                  n5x5=filters[7][2],
                                  pool_planes=filters[7][3],
                                  tmp_name='a5')
        self.inception_b5 = block(in_planes=sum(filters[7]),
                                  n1x1=filters[8][0],
                                  n3x3red=192 if self.layer_cfg is None else self.layer_cfg[24],
                                  n3x3=filters[8][1],
                                  n5x5red1=48 if self.layer_cfg is None else self.layer_cfg[25],
                                  n5x5red2=filters[8][2] if self.layer_cfg is None else self.layer_cfg[26],
                                  n5x5=filters[8][2],
                                  pool_planes=filters[8][3],
                                  tmp_name='b5')

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(sum(filters[-1]), num_classes)

        if self.leader:
            if trans_fusion_info is not None:
                self.trans_fusion_module = TransFusionModule(trans_fusion_info[0], 8, model_num=trans_fusion_info[1])
            else:
                self.trans_fusion_module = TransFusionModule(1024, 8)

        self.reset_parameters()
        self.register_hook()

    def register_hook(self):

        self.extract_layers = ['inception_b3', 'inception_e4', 'inception_b5']

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

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.pre_layers(x)
        # 192 x 32 x 32
        out = self.inception_a3(out)

        # 256 x 32 x 32
        out = self.inception_b3(out)
        # 480 x 32 x 32
        out = self.maxpool1(out)

        # 480 x 16 x 16
        out = self.inception_a4(out)

        # 512 x 16 x 16
        out = self.inception_b4(out)

        # 512 x 16 x 16
        out = self.inception_c4(out)

        # 512 x 16 x 16
        out = self.inception_d4(out)

        # 528 x 16 x 16
        out = self.inception_e4(out)
        # 823 x 16 x 16
        out = self.maxpool2(out)

        # 823 x 8 x 8
        out = self.inception_a5(out)

        # 823 x 8 x 8
        out = self.inception_b5(out)

        if self.leader:
            trans_fusion_output = self.trans_fusion_module(out)

        # 1024 x 8 x 8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.leader:
            return out, trans_fusion_output
        else:
            return out

def googlenet(**kwargs):
    return GoogLeNet(block=Inception, layer_cfg=None, **kwargs)
