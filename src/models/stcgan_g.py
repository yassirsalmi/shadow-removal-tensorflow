#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow_addons.layers import InstanceNormalization

class UnetGenerator(keras.Model):
    """Create a Unet-based generator"""

    def __init__(self, 
                 input_shape,
                 out_channels,
                 ngf=64,
                 num_downs=8,
                 norm_layer=layers.BatchNormalization,
                 use_dropout=False,
                 name='unet_generator'):
        """
        Construct a Unet generator
        
        Args:
            input_shape (tuple): Input shape (H, W, C)
            out_channels (int): Number of output image channels
            ngf (int): Number of filters in the last conv layer
            num_downs (int): Number of downsamplings in UNet
            norm_layer: Normalization layer
            use_dropout (bool): Whether to use dropout
            name (str): Layer name
        """
        super().__init__(name=name)
        
        # Get input channels from shape
        if isinstance(input_shape[-1], int):
            in_channels = input_shape[-1]
        else:
            in_channels = 4  # Default for concatenated RGB + mask
        
        # Construct UNet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, 
            input_nc=None,
            submodule=None, 
            norm_layer=norm_layer,
            innermost=True
        )
        
        # Add intermediate layers with ngf * 8 filters
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, 
                input_nc=None, 
                submodule=unet_block,
                norm_layer=norm_layer, 
                use_dropout=use_dropout
            )
        
        # Gradually reduce number of filters
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, 
            input_nc=None, 
            submodule=unet_block,
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, 
            input_nc=None, 
            submodule=unet_block,
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, 
            input_nc=None, 
            submodule=unet_block,
            norm_layer=norm_layer
        )
        
        # Outermost layer
        self.model = UnetSkipConnectionBlock(
            out_channels, ngf, 
            input_nc=in_channels, 
            submodule=unet_block,
            outermost=True, 
            norm_layer=norm_layer
        )

    def call(self, inputs, training=None):
        """Forward pass"""
        # Handle the case when inputs is a list of tensors
        if isinstance(inputs, (list, tuple)):
            # Concatenate input tensors along the channel axis
            inputs = tf.concat(inputs, axis=-1)
        return self.model(inputs, training=training)


class UnetSkipConnectionBlock(layers.Layer):
    """
    Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=layers.BatchNormalization, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        if input_nc is None:
            input_nc = outer_nc
            
        # Whether to use bias in conv layers (not needed with batch norm)
        use_bias = isinstance(norm_layer, InstanceNormalization)
            
        # Downsampling layers
        self.down_conv = layers.Conv2D(
            inner_nc, kernel_size=4, strides=2, padding='same',
            use_bias=use_bias)
        self.down_norm = norm_layer() if not outermost else None
        self.down_relu = layers.LeakyReLU(0.2)
            
        # Upsampling layers
        if outermost:
            self.up_conv = layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same',
                activation='tanh')
            self.up_norm = None
            self.up_relu = None
        else:
            self.up_conv = layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same',
                use_bias=use_bias)
            self.up_norm = norm_layer()
            self.up_relu = layers.ReLU()
            
        self.submodule = submodule
        self.dropout = layers.Dropout(0.5) if use_dropout else None

    def call(self, x, training=None):
        if self.outermost:
            # No skip connection for outermost layer
            down = self.down_conv(x)
            if self.submodule is not None:
                mid = self.submodule(down, training=training)
            else:
                mid = down
            up = self.up_conv(mid)
            return up
        elif self.innermost:
            # No submodule for innermost layer
            down = self.down_relu(x)
            down = self.down_conv(down)
            if self.down_norm is not None:
                down = self.down_norm(down, training=training)
            up = self.up_relu(down)
            up = self.up_conv(up)
            if self.up_norm is not None:
                up = self.up_norm(up, training=training)
            # Ensure spatial dimensions match exactly
            if x.shape[1:3] != up.shape[1:3]:
                up = tf.image.resize(up, x.shape[1:3], method='nearest')
            return tf.concat([x, up], axis=-1)
        else:
            # Regular case
            down = self.down_relu(x)
            down = self.down_conv(down)
            if self.down_norm is not None:
                down = self.down_norm(down, training=training)
            if self.submodule is not None:
                mid = self.submodule(down, training=training)
            else:
                mid = down
            up = self.up_relu(mid)
            up = self.up_conv(up)
            if self.up_norm is not None:
                up = self.up_norm(up, training=training)
            if self.dropout is not None:
                up = self.dropout(up, training=training)
            # Ensure spatial dimensions match exactly
            if x.shape[1:3] != up.shape[1:3]:
                up = tf.image.resize(up, x.shape[1:3], method='nearest')
            return tf.concat([x, up], axis=-1)
