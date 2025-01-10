#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
M-Net Architectures for Shadow Removal
Inspired by:
Le, H., & Samaras, D. (2019).
Shadow Removal via Shadow Image Decomposition. ICCV.
http://arxiv.org/abs/1908.08628

@InProceedings{Le_2019_ICCV,
    author = {Le, Hieu and Samaras, Dimitris},
    title = {Shadow Removal via Shadow Image Decomposition},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from . import opt_layers
from .skip_connection_layer import SkipConnectionLayer


class MNet(layers.Layer):
    """
    M-Net architecture for shadow removal
    """

    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=True,
                 use_selu=False,
                 activation=None,
                 name='mnet'):
        """
        Initialize M-Net
        
        Args:
            in_channels (int): Number of input image channels
            out_channels (int): Number of output image channels
            ngf (int, optional): Base number of filters
            drop_rate (float, optional): Dropout rate
            no_conv_t (bool, optional): Whether to use upsampling instead of transposed convolution
            use_selu (bool, optional): Whether to use SELU activation
            activation (str or tf.keras.layers.Layer, optional): Final activation layer
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        depth = 4

        # Initial convolution
        self.conv = layers.Conv2D(
            filters=ngf,
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=False
        )

        # Build initial block
        block = SkipConnectionLayer(
            _conv_block((2 ** min(depth-1, 3))*ngf,
                        (2 ** min(depth, 3))*ngf),
            _up_block((2 ** min(depth, 3))*ngf,
                      (2 ** min(depth-1, 3))*ngf, no_conv_t),
            drop_rate=drop_rate
        )

        # Build intermediate blocks
        for i in reversed(range(1, depth-1)):
            features_in = (2 ** min(i, 3)) * ngf
            features_out = (2 ** min(i+1, 3)) * ngf
            block = SkipConnectionLayer(
                _conv_block(features_in, features_out),
                _up_block(2*features_out, features_in, no_conv_t),
                submodule=block,
                drop_rate=drop_rate
            )

        # Final block
        self.block = SkipConnectionLayer(
            _conv_block(ngf, ngf*2),
            _up_block(ngf*4, ngf, no_conv_t),
            submodule=block,
            drop_rate=0
        )

        # Upsample and optional activation
        upsample = opt_layers.get_upsample(no_conv_t, ngf*2, out_channels)
        
        if activation is not None and activation != "none":
            activation_layer = opt_layers.get_activation(activation)
            self.up_conv = tf.keras.Sequential([upsample, activation_layer])
        else:
            self.up_conv = upsample

    def call(self, x, training=False):
        """
        Forward pass through M-Net
        
        Args:
            x (tf.Tensor): Input image tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Processed image tensor
        """
        x = self.conv(x)
        x = self.block(x, training=training)
        return self.up_conv(x, training=training)


class _conv_block(layers.Layer):
    """
    Convolutional block for M-Net
    """
    def __init__(self, in_channels, features, name='conv_block'):
        """
        Initialize convolutional block
        
        Args:
            in_channels (int): Number of input channels
            features (int): Number of output features
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        self.model = tf.keras.Sequential([
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(
                filters=features,
                kernel_size=4,
                strides=2,
                padding='same',
                use_bias=False
            ),
            layers.BatchNormalization(axis=-1)
        ])

    def call(self, x, training=False):
        """
        Forward pass through convolutional block
        
        Args:
            x (tf.Tensor): Input tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tuple: Processed tensor and input tensor
        """
        processed = self.model(x, training=training)
        return processed, x


class _up_block(layers.Layer):
    """
    Upsampling block for M-Net
    """
    def __init__(self, 
                 in_channels, 
                 features, 
                 no_conv_t=True, 
                 name='up_block'):
        """
        Initialize upsampling block
        
        Args:
            in_channels (int): Number of input channels
            features (int): Number of output features
            no_conv_t (bool, optional): Whether to use upsampling instead of transposed convolution
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        
        # Get upsampling method
        upconv = opt_layers.get_upsample(no_conv_t, in_channels, features)
        
        # Create upsampling sequence
        self.model = tf.keras.Sequential([
            layers.LeakyReLU(alpha=0.2),
            upconv,
            layers.BatchNormalization(axis=-1)
        ])

    def call(self, x, link, training=False):
        """
        Forward pass through upsampling block
        
        Args:
            x (tf.Tensor): Input tensor
            link (tf.Tensor): Link tensor from skip connection
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Concatenated tensor
        """
        return tf.concat([self.model(x, training=training), link], axis=-1)
