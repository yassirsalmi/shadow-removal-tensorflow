#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import tensorflow as tf

from stcgan import STCGAN


def main(args):
    """
    Main function to run shadow removal training or inference
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    time_str = time.strftime("%Y%m%d-%H%M%S")
    makedirs(args)
    snapshotargs(args, filename=f"args-{time_str}.json")

    # Set TensorFlow configuration
    tf.config.run_functions_eagerly(False)
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
    
    # Set random seed for reproducibility
    if args.manual_seed != -1:
        set_manual_seed(args.manual_seed)

    # Configure logging
    log_file = os.path.join(
        args.logs, os.path.splitext(__file__)[0]+"-"+time_str+".log")
    set_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    logger.info(args)

    # Initialize STCGAN model
    net = STCGAN(args)

    # Execute tasks
    if "train" in args.tasks:
        net.train(args.epochs)
    if "infer" in args.tasks:
        net.infer()


def set_logger(log_file):
    """
    Configure logging with file and console handlers
    
    Args:
        log_file (str): Path to log file
    """
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    log_formatter = logging.Formatter(
        "%(asctime)s [%(module)s::%(funcName)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        style='%')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def set_manual_seed(manual_seed):
    """
    Set manual random seed for reproducible results
    
    Args:
        manual_seed (int): Seed for random number generators
    """
    # TensorFlow seed setting
    tf.random.set_seed(manual_seed)
    
    # NumPy and Python random seed
    random.seed(manual_seed)
    np.random.seed(manual_seed)


def makedirs(args):
    """
    Create necessary directories for training and inference
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    # Generate directory name based on arguments
    arg_str = f"_lr{args.lr_G:.5f}_"
    if args.D_loss_type == "normal":
        arg_str += ""
    elif args.D_loss_type == "rel":
        arg_str += "Rp"
    else:
        arg_str += "Ra"
    
    if args.D_loss_fn == "standard":
        arg_str += "SGAN"
    else:
        arg_str += "LSGAN"
    
    # Update paths
    args.weights += arg_str
    args.logs += arg_str
    
    # Create directories
    os.makedirs(args.logs, exist_ok=True)
    
    if "train" in args.tasks:
        os.makedirs(args.weights, exist_ok=True)
    
    if "infer" in args.tasks:
        os.makedirs(args.infered, exist_ok=True)
        os.makedirs(os.path.join(args.infered, "shadowless"), exist_ok=True)
        os.makedirs(os.path.join(args.infered, "mask"), exist_ok=True)


def snapshotargs(args, filename="args.json"):
    """
    Save arguments to a JSON file
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
        filename (str, optional): Name of JSON file. Defaults to "args.json".
    """
    args_file = os.path.join(args.logs, filename)
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)


def parse_arguments():
    """
    Parse command-line arguments for shadow removal task
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Training U-Net model for shadow removal"
    )
    parser.add_argument(
        "--tasks",
        help="the task to run (default: %(default)s)",
        required=True, choices=["train", "infer"], type=str, nargs='+',)
    parser.add_argument(
        "--devices",
        help="device for training (default: %(default)s)",
        default=["gpu"], type=str, nargs='+',)
    parser.add_argument(
        "--batch-size",
        help="input batch size for training (default: %(default)d)",
        default=16, type=int,)
    parser.add_argument(
        "--epochs",
        help="number of epochs to train (default: %(default)d)",
        default=100000, type=int,)
    parser.add_argument(
        "--lr-D",
        help="initial learning rate of discriminator (default: %(default).5f)",
        default=0.00002, type=float,)
    parser.add_argument(
        "--lr-G",
        help="initial learning rate of generator (default: %(default).5f)",
        default=0.00005, type=float,)
    parser.add_argument(
        "--decay",
        help=("Decay to apply to lr each cycle. (default: %(default).6f)"
              "(1-decay)^n_iter * lr gives the final lr. "
              "e.g. 0.00002 will lead to .13 of lr after 100k cycles"),
        default=0.00005, type=float)
    parser.add_argument(
        "--workers",
        help="number of workers for data loading (default: %(default)d)",
        default=4, type=int,)
    parser.add_argument(
        "--weights",
        help="folder to save weights (default: %(default)s)",
        default="../weights", type=str,)
    parser.add_argument(
        "--infered",
        help="folder to save infered images (default: %(default)s)",
        default="../infered", type=str,)
    parser.add_argument(
        "--logs",
        help="folder to save logs (default: %(default)s)",
        default="../logs", type=str,)
    parser.add_argument(
        "--data-dir",
        help="root folder with images (default: %(default)s)",
        default="../ISTD_DATASET", type=str,)
    parser.add_argument(
        "--image-size",
        help="target input image size (default: %(default)d)",
        default=256, type=int,)
    parser.add_argument(
        "--aug-scale",
        help=("scale factor range for augmentation "
              "(default: %(default).2f)"),
        default=0.05, type=float,)
    parser.add_argument(
        "--aug-angle",
        help=("rotation range in degrees for augmentation "
              "(default: %(default)d)"),
        default=15, type=int,)
    parser.add_argument(
        "--net-G",
        help="the generator model (default: %(default)s)",
        default="mnet", choices=["unet", "mnet", "denseunet"], type=str,)
    parser.add_argument(
        "--net-D",
        help="the discriminator model (default: %(default)s)",
        default="patchgan", choices=["patchgan"], type=str,)
    parser.add_argument(
        "--load-weights-g1",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-weights-g2",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-weights-d1",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-weights-d2",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--D-loss-fn",
        help="loss funtion of discriminator (default: %(default)s)",
        default="standard", choices=["standard", "leastsquare"], type=str)
    parser.add_argument(
        "--manual-seed",
        help="manual random seed (default: %(default)d)",
        default=-1, type=int)
    parser.add_argument(
        "--D-loss-type",
        help="type of discriminator loss (default: %(default)s)",
        default="normal", choices=["normal", "rel", "adv"], type=str)
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Run main function
    main(args)
