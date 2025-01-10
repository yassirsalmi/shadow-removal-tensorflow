#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2 as cv
import numpy as np
import tensorflow as tf

from tensorflow import transform
from tensorflow.keras import utils


class ISTDDataset(tf.data.Dataset):
    """
    Shadow removal dataset based on ISTD dataset
    Supports flexible data loading and preprocessing
    """
    in_channels: int = 3
    out_channels: int = 3

    def __init__(self, 
                 root_dir,
                 subset,
                 datas: list = ["img", "mask", "target"],
                 transforms=None, 
                 preload=False):
        """
        Initialize ISTD Dataset
        
        Args:
            root_dir (str): Directory with all the images
            subset (str): Dataset subset ('train' or 'test')
            datas (list): Data types to load
            transforms (callable, optional): Optional data transformations
            preload (bool, optional): Whether to preload entire dataset
        """
        super().__init__()
        assert subset in ["train", "test"]
        
        self.root = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, subset, subset + "_A")
        self.mask_dir = os.path.join(root_dir, subset, subset + "_B")
        self.matte_dir = os.path.join(root_dir, subset, subset + "_matte")
        self.target_dir = os.path.join(root_dir, subset, subset + "_C_fixed")

        # Sort files to ensure alignment
        self.img_files = sorted(os.listdir(self.img_dir), 
                                key=lambda f: os.path.splitext(f)[0])
        self.mask_files = sorted(os.listdir(self.mask_dir), 
                                 key=lambda f: os.path.splitext(f)[0])
        self.matte_files = sorted(os.listdir(self.matte_dir), 
                                  key=lambda f: os.path.splitext(f)[0])
        self.target_files = sorted(os.listdir(self.target_dir), 
                                   key=lambda f: os.path.splitext(f)[0])

        # Validate file lists
        assert(len(self.img_files) == len(self.mask_files))
        assert(len(self.img_files) == len(self.matte_files))
        assert(len(self.img_files) == len(self.target_files))

        # Preload data if specified
        self.preload = preload
        if self.preload:
            self.datas = {}
            if "img" in datas:
                self.datas["img"] = [cv.imread(os.path.join(
                    self.img_dir, f), cv.IMREAD_COLOR)
                    for f in self.img_files]
            if "mask" in datas:
                self.datas["mask"] = [cv.imread(os.path.join(
                    self.mask_dir, f), cv.IMREAD_GRAYSCALE)
                    for f in self.mask_files]
            if "matte" in datas:
                self.datas["matte"] = [cv.imread(os.path.join(
                    self.matte_dir, f), cv.IMREAD_GRAYSCALE)
                    for f in self.matte_files]
            if "target" in datas:
                self.datas["target"] = [cv.imread(os.path.join(
                    self.target_dir, f), cv.IMREAD_COLOR)
                    for f in self.target_files]
        else:
            self.datas = datas

    def __getitem__(self, idx):
        """
        Get dataset sample
        
        Args:
            idx (int or tf.Tensor): Sample index
        
        Returns:
            tuple: Dataset sample with filename and tensors
        """
        if isinstance(idx, tf.Tensor):
            idx = idx.numpy()

        # Load sample data
        sample = {}
        if not self.preload:  # Load from disk
            if "img" in self.datas:
                image = cv.imread(os.path.join(
                    self.img_dir, self.img_files[idx]), cv.IMREAD_COLOR)
                sample["img"] = utils.to_float(image)

            if "mask" in self.datas:
                mask = cv.imread(os.path.join(
                    self.mask_dir, self.mask_files[idx]), cv.IMREAD_GRAYSCALE)
                sample["mask"] = utils.to_float(mask)

            if "matte" in self.datas:
                matte = cv.imread(os.path.join(
                    self.matte_dir, self.matte_files[idx]), cv.IMREAD_GRAYSCALE)
                sample["matte"] = utils.to_float(matte)

            if "target" in self.datas:
                target = cv.imread(os.path.join(
                    self.target_dir, self.target_files[idx]), cv.IMREAD_COLOR)
                sample["target"] = utils.to_float(target)
        else:
            for k in self.datas:
                sample[k] = utils.to_float(self.datas[k][idx])

        # Normalize images
        for k in sample:
            sample[k] = (sample[k] - 0.5) * 2

        # Prepare sample list
        sample_list = [sample[k] for k in sorted(sample.keys())]

        # Apply transforms if specified
        if self.transforms is not None:
            sample_list = self.transforms(*sample_list)

        # Add channel dimension for single-channel images
        for i in range(len(sample_list)):
            if sample_list[i].ndim == 2:
                sample_list[i] = sample_list[i][:, :, np.newaxis]

        # Prepare return list
        filename = os.path.splitext(self.img_files[idx])[0]
        return_list = [filename]
        
        # Convert to TensorFlow tensors with channel-first format
        for s in sample_list:
            return_list.append(tf.convert_to_tensor(
                np.transpose(s, (2, 0, 1)), 
                dtype=tf.float32
            ))

        return tuple(return_list)

    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Number of samples in dataset
        """
        return len(self.img_files)

    @classmethod
    def from_tensor_slices(cls, 
                            root_dir, 
                            subset, 
                            batch_size=32, 
                            shuffle=True, 
                            **kwargs):
        """
        Create TensorFlow Dataset from tensor slices
        
        Args:
            root_dir (str): Dataset root directory
            subset (str): Dataset subset
            batch_size (int, optional): Batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
        
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        dataset = cls(root_dir, subset, **kwargs)
        
        # Create TensorFlow dataset from tensor slices
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            [dataset.img_files, 
             dataset.mask_files, 
             dataset.matte_files, 
             dataset.target_files]
        )
        
        # Map dataset to actual data loading
        tf_dataset = tf_dataset.map(
            lambda f1, f2, f3, f4: dataset.__getitem__(
                dataset.img_files.index(f1.numpy().decode())
            )
        )
        
        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
        
        return tf_dataset.batch(batch_size)
