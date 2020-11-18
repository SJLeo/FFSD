import utils.util as util
from options import options

import os

from Tester import Tester

if __name__ == '__main__':

    opt = options.parse()
    util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
    logger = util.get_logger(os.path.join(opt.checkpoints_dir, opt.name, 'logger.log'))

    Tester(opt).test()