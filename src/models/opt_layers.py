#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.layers as layers


def get_activation(key):
    """
    Get activation layer based on key
    
    Args:
        key (str): Activation type ('none', 'sigmoid', 'tanh', 'htanh')
    
    Returns:
        tf.keras.layers.Layer or None: Activation layer
    
    Raises:
        ValueError: If activation key is not recognized
    """
    if key == "sigmoid":
        return layers.Activation('sigmoid')
    elif key == "tanh":
        return layers.Activation('tanh')
    elif key == "htanh":
        return layers.Lambda(lambda x: tf.clip_by_value(x, -1.0, 1.0))
    elif key == "none":
        return None
    else:
        raise ValueError(f"Unsupported activation: {key}")


def get_norm(use_selu: bool, num_features: int):
    """
    Get normalization layer based on SELU flag
    
    Args:
        use_selu (bool): Whether to use SELU activation
        num_features (int): Number of features for normalization
    
    Returns:
        tf.keras.layers.Layer: Normalization layer
    """
    if use_selu:
        return layers.Activation('selu')
    else:
        return tf.keras.Sequential([
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(axis=-1)
        ])


def get_dropout(use_selu: bool, drop_rate):
    """
    Get dropout layer based on SELU flag
    
    Args:
        use_selu (bool): Whether to use SELU activation
        drop_rate (float): Dropout rate
    
    Returns:
        tf.keras.layers.Layer or None: Dropout layer
    """
    if drop_rate == 0:
        return None
    else:
        if use_selu:
            return layers.AlphaDropout(drop_rate)
        else:
            return layers.Dropout(drop_rate)


def get_upsample(use_upsample: bool, in_channels, out_channels):
    """
    Get upsampling layer based on flag
    
    Args:
        use_upsample (bool): Whether to use upsampling or transposed convolution
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        tf.keras.layers.Layer: Upsampling layer
    """
    if use_upsample:
        return tf.keras.Sequential([
            layers.UpSampling2D(size=(2, 2)),
            layers.Conv2D(
                filters=out_channels, 
                kernel_size=3, 
                strides=1, 
                padding='same', 
                use_bias=False
            )
        ])
    else:
        return layers.Conv2DTranspose(
            filters=out_channels, 
            kernel_size=4, 
            strides=2, 
            padding='same', 
            use_bias=False
        )
