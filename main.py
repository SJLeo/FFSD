import utils.util as util
from options import options

import os

from Trainer import Trainer
from Tester import Tester

if __name__ == '__main__':

    opt = options.parse()
    util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
    logger = util.get_logger(os.path.join(opt.checkpoints_dir, opt.name, 'logger.log'))

    if opt.phase == 'train':
        Trainer(opt, logger).train()
    else:
        Tester(opt).test()