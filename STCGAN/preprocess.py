#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from multiprocessing import Pool

import cv2 as cv
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from tensorflow.STCGAN import utils


def polyfit(args: tuple) -> tuple:
    """
    Perform polynomial fitting on image patches
    
    Args:
        args (tuple): Preprocessing arguments
    
    Returns:
        tuple: Polynomial coefficients and patch coordinates
    """
    deg, ksize, r, c, channel, img, gt, weight, w2 = args
    x = img[r:r+ksize, c:c+ksize, channel].ravel()
    y = gt[r:r+ksize, c:c+ksize, channel].ravel()
    w1 = weight[r:r+ksize, c:c+ksize].ravel()
    
    # Use NumPy for polynomial fitting
    coef, _ = np.polynomial.polynomial.polyfit(x, y, deg, full=True, w=w1*w2)
    return (r, c, channel, coef)


def process_images(args: tuple):
    """
    Process images for shadow parameter extraction
    
    Args:
        args (tuple): Image processing arguments
    
    Returns:
        None
    """
    image_dir, target_dir, filename, save_sp, save_img = args
    
    # Read images
    img = cv.imread(os.path.join(image_dir, filename), cv.IMREAD_COLOR)
    target = cv.imread(os.path.join(target_dir, filename), cv.IMREAD_COLOR)

    # Extract shadow parameters
    sp = utils.get_sp(img, target)
    
    # Save shadow parameters
    if save_sp:
        sp_dir = os.path.join(target_dir, os.path.pardir, "sp")
        utils.mkdir(sp_dir)
        np.save(os.path.join(sp_dir, os.path.splitext(filename)[0]), sp)

    # Save restored image
    if save_img:
        img_dir = os.path.join(target_dir, os.path.pardir, "sp_restored_img")
        utils.mkdir(img_dir)
        cv.imwrite(os.path.join(img_dir, filename), utils.apply_sp(img, sp))
    
    return None


def main(args):
    """
    Main preprocessing function
    
    Args:
        args (argparse.Namespace): Preprocessing arguments
    """
    root = args.path
    subset = args.subset
    image_dir = os.path.join(root, subset, subset+"_A")
    target_dir = os.path.join(root, subset, subset+"_C_fixed_official")

    # Get sorted filenames
    filenames = sorted(os.listdir(image_dir))
    print(f"{len(filenames)} files to process", file=sys.stderr)
    
    # Process images
    with Pool() as pool:
        results = list(tqdm(
            pool.imap_unordered(
                process_images, 
                ((image_dir, target_dir, f, args.save_sp, args.save_img) 
                 for f in filenames)
            ), 
            total=len(filenames)
        ))

    # Check for errors
    errors = sum(1 for r in results if r is not None)
    
    if errors > 0:
        print(f"there are {errors} errors", file=sys.stderr)
    else:
        print("completed preprocessing.", file=sys.stderr)


def parse_arguments():
    """
    Parse command-line arguments for preprocessing
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Preprocess shadow removal dataset"
    )
    parser.add_argument(
        "--path", 
        help="Path to ISTD dataset (default: %(default)s)",
        type=str,
        default="../ISTD_DATASET"
    )
    parser.add_argument(
        "--subset", 
        help="the subset to process (default: %(default)s)",
        type=str,
        default="train",
        choices=['train', 'test']
    )
    parser.add_argument(
        "--save-sp", 
        help="whether to save the shadow parameters (default: %(default)s)",
        type=bool,
        nargs='?',
        const=True,
        default=True
    )
    parser.add_argument(
        "--save-img", 
        help="whether to save the image restored with SP (default: %(default)s)",
        type=bool,
        nargs='?',
        const=True,
        default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and run main preprocessing function
    args = parse_arguments()
    main(args)
