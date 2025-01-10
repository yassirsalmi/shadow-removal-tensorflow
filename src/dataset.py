import os
import glob
from typing import List, Optional, Union, Dict, Any

import cv2 as cv
import numpy as np
import tensorflow as tf

import src.utils as utils
import src.transform as transform


AUTOTUNE = tf.data.AUTOTUNE


class ISTDDataset:
    """ISTD dataset loader"""
    
    def __init__(self, data_dir, batch_size=1, image_size=(256, 256)):
        """Initialize dataset"""
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size  
        
        self.img_paths = sorted(glob.glob(os.path.join(data_dir, 'train_A', '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, 'train_B', '*.png')))
        self.target_paths = sorted(glob.glob(os.path.join(data_dir, 'train_C', '*.png')))
        
        assert len(self.img_paths) == len(self.mask_paths) == len(self.target_paths), \
            "Number of images in each directory must be equal"
            
    def _load_and_preprocess(self, img_path, mask_path, target_path):
        """Load and preprocess a single training example"""
        img = tf.io.read_file(img_path)
        mask = tf.io.read_file(mask_path)
        target = tf.io.read_file(target_path)
        
        img = tf.image.decode_png(img, channels=3)
        mask = tf.image.decode_png(mask, channels=1)
        target = tf.image.decode_png(target, channels=3)
        
        img = tf.cast(img, tf.float32) / 127.5 - 1
        mask = tf.cast(mask, tf.float32) / 255.0
        target = tf.cast(target, tf.float32) / 127.5 - 1
        
        img = tf.image.resize(img, self.image_size)
        mask = tf.image.resize(mask, self.image_size)
        target = tf.image.resize(target, self.image_size)
        
        return {'img': img, 'mask': mask, 'target': target}
    
    def get_train_dataset(self):
        """Create training dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((
            self.img_paths,
            self.mask_paths,
            self.target_paths
        ))
        
        dataset = dataset.map(
            lambda x, y, z: self._load_and_preprocess(x, y, z),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def __len__(self):
        """Return number of batches in an epoch"""
        return len(self.img_paths) // self.batch_size


def create_dataset(root_dir: str, 
                  subset: str, 
                  batch_size: int = 32, 
                  shuffle: bool = True, 
                  augment: bool = True,
                  drop_remainder: bool = True,
                  **kwargs) -> tf.data.Dataset:
    """Create dataset with PyTorch-like augmentations"""
    
    dataset = ISTDDataset(root_dir, batch_size)
    
    def load_sample(img_path, mask_path, target_path):
        return dataset._load_and_preprocess(img_path, mask_path, target_path)
    
    ds = tf.data.Dataset.from_tensor_slices((
        dataset.img_paths,
        dataset.mask_paths,
        dataset.target_paths
    ))
    if shuffle:
        ds = ds.shuffle(len(dataset.img_paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y, z: tf.py_function(load_sample, [x, y, z], [tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTOTUNE)
    
    if augment and subset == "train":
        ds = ds.map(lambda x: transform.random_flip(x, prob=0.5))
        ds = ds.map(lambda x: transform.random_crop(x, size=256))
        ds = ds.map(lambda x: transform.random_rotate(x, angle=10))
    
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    
    return ds
