import argparse
import logging
import os


def parse_args():
    parser = argparse.ArgumentParser(description='train DMN+')
    # general
    parser.add_argument('--hidden-size', type=int, default=80, help='hidden size')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    args = parser.parse_args()
    return args


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
