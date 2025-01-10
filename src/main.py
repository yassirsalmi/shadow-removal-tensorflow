"""
Predict a grayscale shadow matte from input image using TensorFlow
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from src.cgan import CGAN


def set_manual_seed(manual_seed):
    """Set manual random seed for reproducible results"""
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    tf.random.set_seed(manual_seed)


def set_logger(log_file):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(module)s::%(funcName)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def makedirs(args):
    """Create necessary directories for logs and checkpoints"""
    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(args.weights, exist_ok=True)


def snapshotargs(args, filename="args.json"):
    """Save arguments to a JSON file"""
    with open(os.path.join(args.logs, filename), 'w') as f:
        json.dump(vars(args), f, indent=4)


def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    """Main entry point"""
    time_str = time.strftime("%Y%m%d-%H%M%S")
    makedirs(args)
    snapshotargs(args, filename="args.json")

    if args.load_args is not None:
        with open(args.load_args, "r") as f:
            arg_dict = json.load(f)
        preserved_args = [
            "load_args", "load_checkpoint", "load_weights_g1", 
            "load_weights_g2", "load_weights_d1", "load_weights_d2", 
            "weights", "logs"
        ]
        for k in preserved_args:
            if k in arg_dict:
                arg_dict.pop(k)
        
        for k, v in arg_dict.items():
            setattr(args, k, v)

    if args.manual_seed != -1:
        set_manual_seed(args.manual_seed)

    log_file = os.path.join(
        args.logs, os.path.splitext(__file__)[0]+"-"+time_str+".log")
    set_logger(log_file)
    logger = logging.getLogger(__name__)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        tf.random.set_seed(args.manual_seed)

    logger.info('Arguments:')
    logger.info(vars(args))

    net = CGAN(args)

    if args.load_checkpoint is not None:
        if not os.path.isfile(args.load_checkpoint):
            print(f"{args.load_checkpoint} is not a file")
        else:
            net.load(path=args.load_checkpoint)

    if 'train' in args.tasks:
        net.train_with_progress_bar(args.epochs)  
    if 'infer' in args.tasks:
        net.infer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shadow Removal with TensorFlow')
    args = parser.parse_args()
    
    main(args)
