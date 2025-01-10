#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers


def weights_init(shape, dtype=None):
    """
    Custom weights initialization for networks
    
    Args:
        shape (tuple): Shape of the weight tensor
        dtype (tf.dtypes.DType, optional): Data type of the tensor
    
    Returns:
        tf.Tensor: Initialized weights
    """
    return tf.random.normal(shape, mean=0.0, stddev=0.02, dtype=dtype)


def get_generator(*args, **kwargs):
    """
    Factory function to create UNet generator
    
    Returns:
        UnetGenerator: Configured UNet generator
    """
    return UnetGenerator(*args, **kwargs)


def get_discriminator(*args, **kwargs):
    """
    Factory function to create PatchGAN discriminator
    
    Returns:
        NLayerDiscriminator: Configured PatchGAN discriminator
    """
    return NLayerDiscriminator(*args, **kwargs)


class UnetGenerator(layers.Layer):
    """
    UNet-based generator with skip connections
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 ngf=64, 
                 num_downs=8, 
                 norm_layer='batch', 
                 use_dropout=False, 
                 name='unet_generator', 
                 **kwargs):
        """
        Construct a UNet generator
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            ngf (int, optional): Base number of filters. Defaults to 64.
            num_downs (int, optional): Number of downsampling layers. Defaults to 8.
            norm_layer (str, optional): Normalization layer type. Defaults to 'batch'.
            use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        """
        super().__init__(name=name, **kwargs)
        
        # Normalization layer selection
        norm_layer_fn = {
            'batch': layers.BatchNormalization,
            'instance': layers.LayerNormalization
        }.get(norm_layer, layers.BatchNormalization)

        # Construct UNet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, 
            input_nc=None,
            submodule=None, 
            norm_layer=norm_layer_fn,
            innermost=True)

        # Add intermediate layers
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, 
                input_nc=None, 
                submodule=unet_block,
                norm_layer=norm_layer_fn, 
                use_dropout=use_dropout)

        # Gradually reduce number of filters
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, 
            input_nc=None, 
            submodule=unet_block,
            norm_layer=norm_layer_fn)
        
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, 
            input_nc=None, 
            submodule=unet_block,
            norm_layer=norm_layer_fn)
        
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, 
            input_nc=None, 
            submodule=unet_block,
            norm_layer=norm_layer_fn)
        
        self.model = UnetSkipConnectionBlock(
            out_channels, ngf, 
            input_nc=in_channels, 
            submodule=unet_block,
            outermost=True, 
            norm_layer=norm_layer_fn)

    def call(self, inputs, training=False):
        """
        Forward pass of the UNet generator
        
        Args:
            inputs (tf.Tensor): Input tensor
            training (bool, optional): Training mode flag
        
        Returns:
            tf.Tensor: Generated output
        """
        return self.model(inputs, training=training)


class UnetSkipConnectionBlock(layers.Layer):
    """
    UNet submodule with skip connections
    """

    def __init__(self, 
                 outer_nc, 
                 inner_nc, 
                 input_nc=None,
                 submodule=None, 
                 outermost=False, 
                 innermost=False,
                 norm_layer=layers.BatchNormalization, 
                 use_dropout=False, 
                 name='unet_skip_block', 
                 **kwargs):
        """
        Construct a UNet submodule with skip connections
        
        Args:
            outer_nc (int): Number of filters in outer conv layer
            inner_nc (int): Number of filters in inner conv layer
            input_nc (int, optional): Number of input channels
            submodule (UnetSkipConnectionBlock, optional): Previous submodules
            outermost (bool, optional): Whether this is the outermost module
            innermost (bool, optional): Whether this is the innermost module
            norm_layer (tf.keras.layers, optional): Normalization layer
            use_dropout (bool, optional): Whether to use dropout
        """
        super().__init__(name=name, **kwargs)
        
        self.outermost = outermost
        input_nc = input_nc or outer_nc

        # Downsampling layers
        downconv = layers.Conv2D(
            inner_nc, 4, strides=2, padding='same', 
            kernel_initializer=weights_init
        )
        downrelu = layers.LeakyReLU(0.2)
        downnorm = norm_layer()

        # Upsampling layers
        uprelu = layers.ReLU()
        upnorm = norm_layer()

        if outermost:
            upconv = layers.Conv2DTranspose(
                outer_nc, 4, strides=2, padding='same', 
                activation='tanh', 
                kernel_initializer=weights_init
            )
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = layers.Conv2DTranspose(
                outer_nc, 4, strides=2, padding='same', 
                kernel_initializer=weights_init
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = layers.Conv2DTranspose(
                outer_nc, 4, strides=2, padding='same', 
                kernel_initializer=weights_init
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [layers.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = tf.keras.Sequential(model)

    def call(self, x, training=False):
        """
        Forward pass of the UNet skip block
        
        Args:
            x (tf.Tensor): Input tensor
            training (bool, optional): Training mode flag
        
        Returns:
            tf.Tensor: Output tensor with skip connection
        """
        if self.outermost:
            return self.model(x, training=training)
        else:
            # Add skip connection
            return tf.concat([x, self.model(x, training=training)], axis=-1)


class NLayerDiscriminator(layers.Layer):
    """
    PatchGAN discriminator
    """

    def __init__(self, 
                 in_channels, 
                 ndf=64, 
                 n_layers=3,
                 norm_layer='batch', 
                 use_sigmoid=False, 
                 name='patchgan_discriminator', 
                 **kwargs):
        """
        Construct a PatchGAN discriminator
        
        Args:
            in_channels (int): Number of input channels
            ndf (int, optional): Base number of filters. Defaults to 64.
            n_layers (int, optional): Number of layers. Defaults to 3.
            norm_layer (str, optional): Normalization layer type. Defaults to 'batch'.
            use_sigmoid (bool, optional): Whether to use sigmoid activation. Defaults to False.
        """
        super().__init__(name=name, **kwargs)
        
        # Normalization layer selection
        norm_layer_fn = {
            'batch': layers.BatchNormalization,
            'instance': layers.LayerNormalization
        }.get(norm_layer, layers.BatchNormalization)

        kw = 4
        padw = 'same'
        
        # Initial layer
        sequence = [
            layers.Conv2D(
                ndf, kw, strides=2, padding=padw, 
                kernel_initializer=weights_init
            ),
            layers.LeakyReLU(0.2)
        ]

        # Progressive layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence.extend([
                layers.Conv2D(
                    ndf * nf_mult, kw, strides=2, padding=padw, 
                    kernel_initializer=weights_init
                ),
                norm_layer_fn(),
                layers.LeakyReLU(0.2)
            ])

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence.extend([
            layers.Conv2D(
                ndf * nf_mult, kw, strides=1, padding=padw, 
                kernel_initializer=weights_init
            ),
            norm_layer_fn(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                1, kw, strides=1, padding=padw, 
                kernel_initializer=weights_init
            )
        ])

        # Optional sigmoid activation
        if use_sigmoid:
            sequence.append(layers.Activation('sigmoid'))

        self.model = tf.keras.Sequential(sequence)

    def call(self, inputs, training=False):
        """
        Forward pass of the discriminator
        
        Args:
            inputs (tf.Tensor): Input tensor
            training (bool, optional): Training mode flag
        
        Returns:
            tf.Tensor: Discriminator output
        """
        return self.model(inputs, training=training)
