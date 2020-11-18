import argparse
import os
import utils.util as util

parser = argparse.ArgumentParser('FFSD')

parser.add_argument('--dataroot', required=True, help='path to images')
parser.add_argument('--dataset', type=str, default='cifar100', help='name of the datasets. Default:cifar10')
parser.add_argument('--name', type=str, default='default', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./experiments', help='models are saved here')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc. default:train')
parser.add_argument('--load_path', type=str, default=None, help='The path of load model. default:None')
parser.add_argument('--model', type=str, default='resnet32', help='chooses which model to use.')


# dataset parameters
parser.add_argument('--train_batch_size', type=int, default=128, help='input batch size. default:128')
parser.add_argument('--test_batch_size', type=int, default=128, help='input batch size. default:128')


# train parameter
parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')


parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--start_epoch', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ... default:1')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs for training. default:100')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate for sgd. default:0.1')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum term of sgd. default:0.9')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay term of sgd. default:1e-4')
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, nargs='+', default=[150, 225], help='the iterval of learn rate. default:150, 225')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='the decay gamma of learn rate. default:0.1')

# mutula learing
parser.add_argument('--model_num', type=int, default=2, help='the number of models for online kd. default:2.')
parser.add_argument('--temperature', type=int, default=2, help='the temperature for online kd. default:2.')
parser.add_argument('--lambda_diversity', type=float, default=1e-5, help='the coefficient for model diversity loss. default:1e-5')
parser.add_argument('--lambda_fusion', type=float, default=10.0, help='the coefficient of distilling the fusion knowledge for the student leader. default:10.0')
parser.add_argument('--lambda_self_distillation', type=float, default=1000.0, help='the coefficient for self-distillation train loss. default:1000.0')


def print_options(opt, parser):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / config.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'config.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def parse():

    opt = parser.parse_args()
    print_options(opt, parser)
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)

    return opt