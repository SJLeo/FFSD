import utils.util as util
from options import options

import os

from DistributeImageNetTrainer import Trainer

if __name__ == '__main__':

    opt = options.parse()
    util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
    logger = util.get_logger(os.path.join(opt.checkpoints_dir, opt.name, 'logger.log'))

    Trainer(opt, logger).train()