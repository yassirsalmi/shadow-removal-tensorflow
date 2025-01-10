"""
Advanced visualization utilities for shadow removal project
"""

import os
from typing import List, Union, Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: Normalized image
    """
    min_val, max_val = image.min(), image.max()
    return (image - min_val) / (max_val - min_val) if max_val > min_val else image


def denormalize_image(image: np.ndarray, 
                      mean: Optional[List[float]] = None, 
                      std: Optional[List[float]] = None) -> np.ndarray:
    """
    Denormalize image
    
    Args:
        image (np.ndarray): Normalized image
        mean (list, optional): Mean values for each channel
        std (list, optional): Standard deviation values for each channel
    
    Returns:
        np.ndarray: Denormalized image
    """
    if mean is not None and std is not None:
        return image * np.array(std) + np.array(mean)
    return normalize_image(image)


def plot_images(images: Union[List[np.ndarray], np.ndarray], 
                titles: Optional[List[str]] = None, 
                figsize: tuple = (15, 5), 
                cmap: Optional[str] = None):
    """
    Plot multiple images in a single figure
    
    Args:
        images (list or np.ndarray): Images to plot
        titles (list, optional): Titles for each image
        figsize (tuple, optional): Figure size
        cmap (str, optional): Colormap for plotting
    """
    if not isinstance(images, list):
        images = [images]
    
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        img = normalize_image(img)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()


def save_image_grid(images: Union[List[np.ndarray], np.ndarray], 
                    output_path: str, 
                    grid_size: Optional[tuple] = None, 
                    padding: int = 5):
    """
    Save images in a grid
    
    Args:
        images (list or np.ndarray): Images to save
        output_path (str): Path to save grid image
        grid_size (tuple, optional): Grid dimensions
        padding (int, optional): Padding between images
    """
    if not isinstance(images, list):
        images = [images]
    
    images = [normalize_image(img) for img in images]
    
    if grid_size is None:
        grid_size = (int(np.ceil(np.sqrt(len(images)))), 
                     int(np.ceil(np.sqrt(len(images)))))
    
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    grid_height = grid_size[0] * (max_height + padding)
    grid_width = grid_size[1] * (max_width + padding)
    
    grid = np.ones((grid_height, grid_width, 3), dtype=np.float32)
    
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        start_h = row * (max_height + padding)
        start_w = col * (max_width + padding)
        
        resized_img = cv.resize(img, (max_width, max_height))
        
        grid[start_h:start_h+max_height, start_w:start_w+max_width] = resized_img
    
    plt.imsave(output_path, grid)


def log_images_to_tensorboard(
    writer: tf.summary.SummaryWriter,
    tag: str,
    images: Union[List[np.ndarray], np.ndarray],
    step: int,
    max_outputs: int = 4
):
    """
    Log images to TensorBoard
    
    Args:
        writer (tf.summary.SummaryWriter): TensorBoard summary writer
        tag (str): Logging tag
        images (list or np.ndarray): Images to log
        step (int): Training step
        max_outputs (int, optional): Maximum number of images to log
    """
    if not isinstance(images, list):
        images = [images]
    
    images = [normalize_image(img) for img in images]
    images = images[:max_outputs]
    
    images = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in images]
    
    with writer.as_default():
        tf.summary.image(tag, images, step=step, max_outputs=max_outputs)


def visualize_shadow_removal(
    input_image: np.ndarray,
    shadow_mask: np.ndarray,
    output_image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None
):
    """
    Visualize shadow removal results
    
    Args:
        input_image (np.ndarray): Original input image
        shadow_mask (np.ndarray): Shadow mask
        output_image (np.ndarray): Shadow-removed image
        ground_truth (np.ndarray, optional): Ground truth image
        output_dir (str, optional): Directory to save visualization
    """
    input_image = normalize_image(input_image)
    shadow_mask = normalize_image(shadow_mask)
    output_image = normalize_image(output_image)
    
    images = [input_image, shadow_mask, output_image]
    titles = ['Input Image', 'Shadow Mask', 'Shadow Removed']
    
    if ground_truth is not None:
        ground_truth = normalize_image(ground_truth)
        images.append(ground_truth)
        titles.append('Ground Truth')
    
    plot_images(images, titles=titles)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, (img, title) in enumerate(zip(images, titles)):
            output_path = os.path.join(output_dir, f'{title.lower().replace(" ", "_")}.png')
            plt.imsave(output_path, img)


def main():
    """
    Example usage of visualization functions
    """
    sample_image = np.random.rand(256, 256, 3)
    sample_mask = np.random.rand(256, 256)
    sample_output = np.random.rand(256, 256, 3)
    
    visualize_shadow_removal(
        sample_image, 
        sample_mask, 
        sample_output, 
        output_dir='./visualization_samples'
    )
    
    save_image_grid(
        [sample_image, sample_mask, sample_output], 
        './visualization_samples/image_grid.png'
    )


if __name__ == "__main__":
    main()
