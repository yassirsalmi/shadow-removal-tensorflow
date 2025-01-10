#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BEGAN (Boundary Equilibrium Generative Adversarial Networks) 
Image Translation Model
"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from . import opt_layers


def conv_block(in_dim, out_dim, use_selu=False):
    """
    Create a convolutional downsampling block
    
    Args:
        in_dim (int): Input channel dimension
        out_dim (int): Output channel dimension
        use_selu (bool, optional): Whether to use SELU activation
    
    Returns:
        tf.keras.Sequential: Convolutional block
    """
    return tf.keras.Sequential([
        layers.Conv2D(out_dim, kernel_size=3, strides=1, padding='same'),
        opt_layers.get_norm(use_selu, out_dim),
        layers.MaxPool2D(pool_size=2, strides=2)
    ])


def deconv_block(in_dim, out_dim, use_selu=False):
    """
    Create a deconvolutional upsampling block
    
    Args:
        in_dim (int): Input channel dimension
        out_dim (int): Output channel dimension
        use_selu (bool, optional): Whether to use SELU activation
    
    Returns:
        tf.keras.Sequential: Deconvolutional block
    """
    return tf.keras.Sequential([
        layers.Conv2D(out_dim, kernel_size=3, strides=1, padding='same'),
        opt_layers.get_norm(use_selu, out_dim),
        layers.UpSampling2D(size=(2, 2))
    ])


class BEGAN(layers.Layer):
    """
    BEGAN (Boundary Equilibrium Generative Adversarial Networks) 
    Image Translation Model
    """

    def __init__(self, 
                 in_channels, 
                 out_channels=None,
                 ndf=64,
                 n_layers=3,
                 use_selu=False,
                 use_sigmoid=False,
                 name='began'):
        """
        Initialize BEGAN model
        
        Args:
            in_channels (int): Number of input image channels
            out_channels (int, optional): Number of output image channels
            ndf (int, optional): Base number of discriminator filters
            n_layers (int, optional): Number of layers
            use_selu (bool, optional): Whether to use SELU activation
            use_sigmoid (bool, optional): Whether to use sigmoid activation
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        
        # Input convolution
        self.in_conv = layers.Conv2D(
            filters=ndf, 
            kernel_size=3, 
            strides=1, 
            padding='same'
        )
        self.in_norm = opt_layers.get_norm(use_selu, ndf)
        
        # Downsampling layers
        self.downsamples = []
        prev_channels = ndf
        for n in range(1, n_layers):
            down_block = conv_block(prev_channels, ndf*n, use_selu)
            self.downsamples.append(down_block)
            prev_channels = ndf*n
        
        # Bottleneck layers
        self.bottleneck = [
            layers.Conv2D(ndf, kernel_size=3, strides=1, padding='same'),
            layers.Conv2D(ndf, kernel_size=3, strides=1, padding='same')
        ]
        
        # Decoder layers
        self.decoders = [deconv_block(ndf, ndf, use_selu)]
        for n in reversed(range(1, n_layers-1)):
            self.decoders.append(deconv_block(2*ndf, ndf, use_selu))
        
        # Output convolution
        out_channels = out_channels or in_channels
        out_conv = layers.Conv2D(
            filters=out_channels, 
            kernel_size=3, 
            strides=1, 
            padding='same'
        )
        
        # Final activation
        if use_sigmoid:
            out_act = layers.Activation('sigmoid')
        else:
            out_act = layers.Activation('tanh')
        
        self.out_conv = tf.keras.Sequential([out_conv, out_act])

    def call(self, x, training=False):
        """
        Forward pass through the BEGAN model
        
        Args:
            x (tf.Tensor): Input image tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Generated or transformed image
        """
        # Input convolution
        x = self.in_conv(x)
        x = self.in_norm(x, training=training)
        
        # Downsampling
        for encoder in self.downsamples:
            x = encoder(x)
        
        # Bottleneck
        x = self.bottleneck[0](x)
        x = self.bottleneck[1](x)
        
        # Decoding
        y = x
        for i, decoder in enumerate(self.decoders):
            if i < len(self.decoders) - 1:
                # Upsample and concatenate
                upsampled_x = tf.image.resize(
                    x, 
                    size=x.shape[1:3] * (2 ** (i+1)), 
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )
                y = decoder(y)
                y = tf.concat([upsampled_x, y], axis=-1)
            else:
                y = decoder(y)
        
        return self.out_conv(y)
