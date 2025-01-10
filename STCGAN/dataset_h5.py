#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import tensorflow as tf

from tensorflow import transform


class ISTDDataset(tf.data.Dataset):
    """
    Shadow removal dataset based on ISTD dataset using HDF5 file
    """
    in_channels: int = 3
    out_channels: int = 3

    # B, G, R
    mean = [0.54, 0.57, 0.57]
    std = [0.14, 0.14, 0.14]

    def __init__(self,
                 file,
                 subset="train",
                 transforms=None):
        """
        Initialize HDF5 dataset
        
        Args:
            file (str): Path to HDF5 file
            subset (str, optional): Dataset subset. Defaults to "train".
            transforms (callable, optional): Optional transformations
        """
        super().__init__()
        assert subset in ["train", "test"]
        self.data_set = h5py.File(file, 'r')[subset]
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx (int or tf.Tensor): Sample index
        
        Returns:
            tuple: Dataset sample with filename, input image, target image, and shadow map
        """
        if isinstance(idx, tf.Tensor):
            idx = idx.numpy()

        # Read images and metadata
        input_img = self.data_set["input_img"][idx]
        target_img = self.data_set["target_img"][idx]
        sp = self.data_set["sp"][idx]
        filename = self.data_set["filename"][idx]

        # Normalize images
        normalize = transform.Normalize(ISTDDataset.mean, ISTDDataset.std)
        input_img, target_img = normalize(input_img, target_img)

        # Apply optional transforms
        if self.transforms is not None:
            input_img, sp, target_img = \
                self.transforms(input_img, sp, target_img)

        # Convert to TensorFlow tensors
        input_img_tensor = tf.convert_to_tensor(
            np.transpose(input_img, (2, 0, 1)), 
            dtype=tf.float32
        )
        target_img_tensor = tf.convert_to_tensor(
            np.transpose(target_img, (2, 0, 1)), 
            dtype=tf.float32
        )
        sp_tensor = tf.convert_to_tensor(
            np.transpose(sp, (2, 0, 1)), 
            dtype=tf.float32
        )

        return (filename,
                input_img_tensor,
                target_img_tensor,
                sp_tensor)

    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Number of samples in dataset
        """
        return self.data_set["filename"].shape[0]

    @classmethod
    def from_tensor_slices(cls, 
                            file, 
                            subset="train", 
                            batch_size=32, 
                            shuffle=True, 
                            **kwargs):
        """
        Create TensorFlow Dataset from HDF5 file
        
        Args:
            file (str): Path to HDF5 file
            subset (str, optional): Dataset subset. Defaults to "train".
            batch_size (int, optional): Batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
        
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        dataset = cls(file, subset)
        
        # Create TensorFlow dataset from tensor slices
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            (dataset.data_set["filename"][:],
             dataset.data_set["input_img"][:],
             dataset.data_set["target_img"][:],
             dataset.data_set["sp"][:])
        )
        
        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
        
        return tf_dataset.batch(batch_size)
