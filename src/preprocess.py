import argparse
import os
import sys
from multiprocessing import Pool
from typing import List, Tuple, Optional

import tensorflow as tf
import numpy as np
import cv2 as cv
from tqdm.auto import tqdm

from . import utils


def polyfit(args: Tuple) -> Tuple:
    """
    Perform polynomial fitting on image patches
    
    Args:
        args (tuple): Contains parameters for polynomial fitting
    
    Returns:
        tuple: Patch coordinates, channel, and polynomial coefficients
    """
    deg, ksize, r, c, channel, img, gt, weight, w2 = args
    x = img[r:r+ksize, c:c+ksize, channel].ravel()
    y = gt[r:r+ksize, c:c+ksize, channel].ravel()
    w1 = weight[r:r+ksize, c:c+ksize].ravel()
    
    coef, _ = np.polynomial.polynomial.polyfit(x, y, deg, full=True, w=w1*w2)
    return (r, c, channel, coef)


def process_images(args: Tuple[str, str, str, bool, bool]) -> Optional[int]:
    """
    Process images for shadow parameter extraction and restoration
    
    Args:
        args (tuple): Contains image directory, target directory, filename, 
                      and flags for saving shadow parameters and restored images
    
    Returns:
        Optional[int]: Error code if processing fails, None otherwise
    """
    image_dir, target_dir, filename, save_sp, save_img = args
    
    try:
        img = cv.imread(os.path.join(image_dir, filename), cv.IMREAD_COLOR)
        target = cv.imread(os.path.join(target_dir, filename), cv.IMREAD_COLOR)

        sp = utils.get_sp(img, target)
        
        if save_sp:
            sp_dir = os.path.join(target_dir, os.path.pardir, "sp")
            utils.mkdir(sp_dir)
            np.save(os.path.join(sp_dir, os.path.splitext(filename)[0]), sp)

        if save_img:
            img_dir = os.path.join(target_dir, os.path.pardir, "sp_restored_img")
            utils.mkdir(img_dir)
            cv.imwrite(os.path.join(img_dir, filename), utils.apply_sp(img, sp))
        
        return None
    except Exception as e:
        print(f"Error processing {filename}: {e}", file=sys.stderr)
        return 1


def main(args):
    """
    Main preprocessing function for shadow removal dataset
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    root = args.path
    subset = args.subset
    image_dir = os.path.join(root, subset, subset+"_A")
    target_dir = os.path.join(root, subset, subset+"_C_fixed_official")

    filenames = sorted(os.listdir(image_dir))
    print(f"{len(filenames)} files to process", file=sys.stderr)
    
    process_args = [
        (image_dir, target_dir, f, args.save_sp, args.save_img) 
        for f in filenames
    ]
    
    with Pool() as pool:
        results = list(tqdm(
            pool.imap(process_images, process_args), 
            total=len(filenames)
        ))

    errors = sum(1 for result in results if result is not None)
    
    if errors > 0:
        print(f"There are {errors} errors", file=sys.stderr)
    else:
        print("Completed preprocessing.", file=sys.stderr)
    
    return


def parse_arguments() -> argparse.Namespace:
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
        help="The subset to process (default: %(default)s)",
        type=str,
        default="train",
        choices=['train', 'test']
    )
    parser.add_argument(
        "--save-sp", 
        help="Whether to save the shadow parameters (default: %(default)s)",
        type=bool,
        nargs='?',
        const=True,
        default=True
    )
    parser.add_argument(
        "--save-img", 
        help="Whether to save the image restored with SP (default: %(default)s)",
        type=bool,
        nargs='?',
        const=True,
        default=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
