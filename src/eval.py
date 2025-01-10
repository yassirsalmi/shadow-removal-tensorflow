"""
Evaluate shadow removal model performance
"""

import argparse
import json
import logging
import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from skimage import io, color, transform, util
from tqdm.auto import tqdm


def str2bool(v):
    """
    Convert string to boolean
    
    Args:
        v: Input value
    
    Returns:
        bool: Converted boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def snapshotargs(args, filename="args.json"):
    """
    Save arguments to JSON file
    
    Args:
        args: Argument namespace
        filename: Output filename
    """
    with open(filename, "w") as f:
        json.dump(vars(args), f, indent=4)


def set_logger(log_file=None, level=logging.INFO):
    """
    Configure logging
    
    Args:
        log_file: Optional log file path
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def MSE(img1, img2):
    """
    Mean Squared Error
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        float: MSE value
    """
    return np.mean((img1 - img2) ** 2)


def MAE(img1, img2, mask=None):
    """
    Mean Absolute Error
    
    Args:
        img1: First image
        img2: Second image
        mask: Optional mask for computation
    
    Returns:
        float: MAE value
    """
    diff = np.abs(img1 - img2)
    return np.mean(diff[mask]) if mask is not None else np.mean(diff)


def RMSE(img1, img2, mask=None):
    """
    Root Mean Square Error
    
    Args:
        img1: First image
        img2: Second image
        mask: Optional mask for computation
    
    Returns:
        float: RMSE value
    """
    return np.sqrt(MSE(img1[mask], img2[mask]) if mask is not None else MSE(img1, img2))


def PSNR(img1, img2):
    """
    Peak Signal-to-Noise Ratio
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        float: PSNR value
    """
    mse = MSE(img1, img2)
    max_pixel = np.max(img1)
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def SSIM(img1, img2):
    """
    Structural Similarity Index
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        float: SSIM value
    """
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, multichannel=True)


def all_metrics(dir1, dir2, size=None, maskdir=None):
    """
    Compute multiple image quality metrics
    
    Args:
        dir1: First directory with images
        dir2: Second directory with images
        size: Optional resize dimensions
        maskdir: Optional directory with mask images
    
    Returns:
        dict: Computed metrics
    """
    files = os.listdir(dir1)
    rmses, maes = [], []
    rmses_nonshadow, maes_nonshadow = [], []
    pixels, pixels_nonshadow = [], []
    psnrs, ssims = [], []

    for f in tqdm(files):
        img1 = util.img_as_float32(io.imread(os.path.join(dir1, f)))
        img2 = transform.resize(
            util.img_as_float32(io.imread(os.path.join(dir2, f))),
            img1.shape, mode="edge", anti_aliasing=False)

        if maskdir is not None:
            mask = transform.resize(
                io.imread(os.path.join(maskdir, f), as_gray=True),
                (img1.shape[:2]), mode="edge")
        else:
            mask = np.ones((img1.shape[0], img1.shape[1]), dtype=bool)

        if size is not None:
            img1_resized = transform.resize(
                img1, (size, size), mode="edge", anti_aliasing=False)
            img2_resized = transform.resize(
                img2, (size, size), mode="edge", anti_aliasing=False)
            mask_resized = util.img_as_bool(transform.resize(
                mask, (size, size), mode="edge"))
        else:
            img1_resized, img2_resized = img1, img2
            mask_resized = util.img_as_bool(mask)

        lab_img1 = color.rgb2lab(img1_resized)
        lab_img2 = color.rgb2lab(img2_resized)

        rmses.append(RMSE(lab_img1, lab_img2, mask_resized))
        maes.append(MAE(lab_img1, lab_img2, mask_resized))
        pixels.append(np.count_nonzero(mask_resized))

        mask_nonshadow = np.logical_not(mask_resized)
        rmses_nonshadow.append(RMSE(lab_img1, lab_img2, mask_nonshadow))
        maes_nonshadow.append(MAE(lab_img1, lab_img2, mask_nonshadow))
        pixels_nonshadow.append(np.count_nonzero(mask_nonshadow))

        psnrs.append(PSNR(img1_resized, img2_resized))
        ssims.append(SSIM(img1_resized, img2_resized))

    return {
        "rmse": np.mean(rmses),
        "mae": np.mean(maes),
        "rmse_nonshadow": np.mean(rmses_nonshadow),
        "mae_nonshadow": np.mean(maes_nonshadow),
        "pixels": np.mean(pixels),
        "pixels_nonshadow": np.mean(pixels_nonshadow),
        "psnr": np.mean(psnrs),
        "ssim": np.mean(ssims)
    }


def main(args):
    """
    Main evaluation function
    
    Args:
        args: Command-line arguments
    """
    snapshotargs(args, filename="args.json")

    set_logger(args.logfile)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    logger.info(args)

    errors = all_metrics(
        args.dir1, 
        args.dir2, 
        size=args.image_size, 
        maskdir=args.maskdir
    )

    for k, v in errors.items():
        logger.info(f"{k}: {v}")


def cli():
    """
    Command-line interface for evaluation
    """
    parser = argparse.ArgumentParser(description="Evaluate shadow removal model")
    parser.add_argument("dir1", type=str, help="First directory with images")
    parser.add_argument("dir2", type=str, help="Second directory with images")
    parser.add_argument(
        "--image-size", 
        type=int, 
        default=None, 
        help="Resize images to specified size"
    )
    parser.add_argument(
        "--maskdir", 
        type=str, 
        default=None, 
        help="Directory with mask images"
    )
    parser.add_argument(
        "--logfile", 
        type=str, 
        default=None, 
        help="Path to log file"
    )
    parser.add_argument(
        "--verbose", 
        type=str2bool, 
        nargs='?', 
        const=True, 
        default=False, 
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
