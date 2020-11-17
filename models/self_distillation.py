import torch.nn as nn

class SelfDistillationModule(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(SelfDistillationModule, self).__init__()

        self.convtranpose = nn.ConvTranspose2d(in_channels=input_channel,
                                               out_channels=output_channel,
                                               kernel_size=3, stride=2,
                                               padding=1, output_padding=1, bias=False)
        self.norm = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(True)

    def forward(self, x):

        x = self.convtranpose(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class SelfDistillationModel(nn.Module):

    def __init__(self, input_channel, layer_num):
        super(SelfDistillationModel, self).__init__()

        self.layer_num = layer_num
        self.total_feature_maps = {}
        output_channel = int(input_channel / 2)

        for i in range(layer_num):
            setattr(self, 'layer%d' % i, SelfDistillationModule(input_channel, output_channel))
            input_channel = output_channel
            output_channel = int(input_channel / 2)

        self.register_hook()

    def forward(self, x):

        for i in range(self.layer_num):
            x = getattr(self, 'layer%d' % i)(x)

        return x

    def register_hook(self):

        self.extract_layers = [('layer%d' % i) for i in range(self.layer_num)]

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name+str(output.device)] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)

class DIYSelfDistillationModel(nn.Module):

    def __init__(self, channel_nums, layer_num):
        super(DIYSelfDistillationModel, self).__init__()

        self.layer_num = layer_num
        self.total_feature_maps = {}

        for i in range(layer_num):
            setattr(self, 'layer%d' % i, SelfDistillationModule(channel_nums[i], channel_nums[i+1]))

        self.register_hook()

    def forward(self, x):

        for i in range(self.layer_num):
            x = getattr(self, 'layer%d' % i)(x)

        return x

    def register_hook(self):

        self.extract_layers = [('layer%d' % i) for i in range(self.layer_num)]

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                # maps[name] = output.pow(2).mean(1, keepdim=True)
                maps[name+str(output.device)] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self, self.total_feature_maps, self.extract_layers)