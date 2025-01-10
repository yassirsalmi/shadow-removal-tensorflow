#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PatchGAN Discriminator for Image-to-Image Translation
Inspired by:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from . import opt_layers


class PatchGAN(layers.Layer):
    """
    PatchGAN Discriminator for Image-to-Image Translation
    
    Classifies image patches as real or fake
    """

    def __init__(self, 
                 in_channels,
                 ndf=64,
                 n_layers=3,
                 use_selu=False,
                 use_sigmoid=False,
                 name='patchgan'):
        """
        Initialize PatchGAN Discriminator
        
        Args:
            in_channels (int): Number of input image channels
            ndf (int, optional): Base number of discriminator filters
            n_layers (int, optional): Number of discriminator layers
            use_selu (bool, optional): Whether to use SELU activation
            use_sigmoid (bool, optional): Whether to use sigmoid activation
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        
        # Build discriminator sequence
        sequence = [
            layers.Conv2D(
                filters=ndf, 
                kernel_size=4, 
                strides=2, 
                padding='same',
                input_shape=(None, None, in_channels)
            ),
            layers.LeakyReLU(alpha=0.2)
        ]
        
        prev_channels = ndf
        for n in range(1, n_layers):
            if n < 4:
                sequence.extend(self._block(prev_channels, prev_channels*2, use_selu))
                prev_channels *= 2
            else:
                sequence.extend(self._block(prev_channels, prev_channels, use_selu))

        out_channels = prev_channels*2 if n_layers < 4 else prev_channels
        
        # Additional convolution layers
        sequence.extend([
            layers.Conv2D(
                filters=out_channels, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                use_bias=False
            ),
            opt_layers.get_norm(use_selu=use_selu, num_features=out_channels)
        ])

        # Final convolution to single channel
        sequence.append(
            layers.Conv2D(
                filters=1, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                use_bias=False
            )
        )

        # Optional sigmoid activation
        if use_sigmoid:
            sequence.append(layers.Activation('sigmoid'))

        # Create sequential model
        self.model = tf.keras.Sequential(sequence)

    def call(self, x, training=False):
        """
        Forward pass through PatchGAN
        
        Args:
            x (tf.Tensor): Input image tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Discriminator output
        """
        return self.model(x, training=training)

    def _block(self, in_channels, out_channels=None, use_selu=False):
        """
        Create a discriminator block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int, optional): Number of output channels
            use_selu (bool, optional): Whether to use SELU activation
        
        Returns:
            list: Block layers
        """
        if out_channels is None:
            out_channels = in_channels * 2
        
        return [
            layers.Conv2D(
                filters=out_channels, 
                kernel_size=4, 
                strides=2, 
                padding='same', 
                use_bias=False
            ),
            opt_layers.get_norm(use_selu=use_selu, num_features=out_channels)
        ]
