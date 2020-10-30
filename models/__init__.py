import importlib
import re

def create_model(opt=None, leader=False, trans_fusion_info=None):

    arch = re.sub('\d', '', opt.model)

    if opt.dataset == 'imagenet':
        arch += '_imagenet'
        num_classes = 1000
    elif opt.dataset == 'cifar100':
        arch += '_cifar'
        num_classes = 100
    else:
        arch += '_cifar'
        num_classes = 10

    model_filename = f'models.{arch}'
    model_lib = importlib.import_module(model_filename)

    model_cls = None
    for name, cls in model_lib.__dict__.items():
        if name.lower() == opt.model.lower():
            model_cls = cls

    model = model_cls(num_classes=num_classes, leader=leader, trans_fusion_info=trans_fusion_info)
    return model