import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='train DMN+')
    # general
    parser.add_argument('--hidden-size', type=int, default=80, help='hidden size')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size')
    args = parser.parse_args()
    return args


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
