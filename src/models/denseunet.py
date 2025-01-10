#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dense UNet for Image Segmentation and Translation
Inspired by:
https://github.com/mateuszbuda/brain-segmentation-pytorch
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape
         features automatically extracted by a deep learning algorithm},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal={Computers in Biology and Medicine},
  volume={109},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2019.05.002}
}
"""

import tensorflow as tf
import tensorflow.keras.layers as layers

from .skip_connection_layer import SkipConnectionLayer


class DenseUNet(layers.Layer):
    """
    Dense UNet with flexible architecture and skip connections
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ngf=48,
                 drop_rate=0,
                 no_conv_t=False,
                 activation=None,
                 name='dense_unet'):
        """
        Initialize Dense UNet
        
        Args:
            in_channels (int): Number of input image channels
            out_channels (int): Number of output image channels
            ngf (int, optional): Base number of filters
            drop_rate (float, optional): Dropout rate
            no_conv_t (bool, optional): Whether to use upsampling instead of transposed convolution
            activation (tf.keras.layers.Layer, optional): Final activation layer
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        
        depth = 5
        n_composite_layers = 2
        growth_rate = ngf // n_composite_layers

        # Input convolution
        in_conv = layers.Conv2D(
            filters=ngf, 
            kernel_size=1, 
            strides=1, 
            padding='valid', 
            use_bias=False
        )

        # Bottleneck block
        block = self._bottleneck(
            ngf, layers=3*n_composite_layers, growth_rate=growth_rate)

        # Build UNet layers with skip connections
        for i in reversed(range(depth)):
            block = SkipConnectionLayer(
                _conv_block(ngf, n_composite_layers, growth_rate),
                _up_block(ngf*4, ngf*2, n_composite_layers,
                          growth_rate, no_conv_t),
                submodule=block,
                drop_rate=drop_rate if i > 0 else 0
            )

        # Output convolution
        out_conv = layers.Conv2D(
            filters=out_channels, 
            kernel_size=1, 
            strides=1, 
            use_bias=False
        )

        # Build final sequence
        sequence = [in_conv, block, out_conv]
        if activation is not None and activation != "none":
            sequence.append(activation)

        self.model = tf.keras.Sequential(sequence)

    def call(self, x, training=False):
        """
        Forward pass through Dense UNet
        
        Args:
            x (tf.Tensor): Input image tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Processed image tensor
        """
        return self.model(x, training=training)

    @staticmethod
    def _trans_down(in_channels, out_channels=None, drop_rate=0.01):
        """
        Create downsampling transition block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int, optional): Number of output channels
            drop_rate (float, optional): Dropout rate
        
        Returns:
            tf.keras.Sequential: Downsampling transition block
        """
        if out_channels is None:
            out_channels = in_channels // 2
        
        block = [
            layers.BatchNormalization(axis=-1),
            layers.Conv2D(
                filters=out_channels, 
                kernel_size=1, 
                strides=1, 
                padding='valid', 
                use_bias=False
            )
        ]
        
        if drop_rate > 0:
            block.append(layers.Dropout(drop_rate))
        
        block.append(layers.AveragePooling2D(pool_size=2))
        
        return tf.keras.Sequential(block)

    @staticmethod
    def _trans_up(in_channels, out_channels=None, no_conv_t=False):
        """
        Create upsampling transition block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int, optional): Number of output channels
            no_conv_t (bool, optional): Whether to use upsampling instead of transposed convolution
        
        Returns:
            tf.keras.layers.Layer: Upsampling transition block
        """
        if out_channels is None:
            out_channels = in_channels // 4

        if no_conv_t:
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
                kernel_size=2, 
                strides=2, 
                use_bias=False
            )

    @staticmethod
    def _bottleneck(in_channels, layers=8, growth_rate=8):
        """
        Create bottleneck dense block
        
        Args:
            in_channels (int): Number of input channels
            layers (int, optional): Number of layers in dense block
            growth_rate (int, optional): Growth rate for dense block
        
        Returns:
            _dense_block: Bottleneck dense block
        """
        return DenseUNet._dense_block(
            in_channels, layers=layers, growth_rate=growth_rate, drop_rate=0)

    class _dense_block(layers.Layer):
        """
        Dense block implementation
        """
        def __init__(self, in_channels, layers=4,
                     growth_rate=8, drop_rate=0.01, name='dense_block'):
            """
            Initialize dense block
            
            Args:
                in_channels (int): Number of input channels
                layers (int, optional): Number of layers in dense block
                growth_rate (int, optional): Growth rate for dense block
                drop_rate (float, optional): Dropout rate
                name (str, optional): Layer name
            """
            super().__init__(name=name)
            self.composite_layers = [
                self._composite(
                    in_channels+i*growth_rate, growth_rate, drop_rate)
                for i in range(layers)
            ]

        def call(self, x, training=False):
            """
            Forward pass through dense block
            
            Args:
                x (tf.Tensor): Input tensor
                training (bool, optional): Whether in training mode
            
            Returns:
                tf.Tensor: Output tensor
            """
            for composite_layer in self.composite_layers:
                y = x
                x = composite_layer(x, training=training)
                x = tf.concat([x, y], axis=-1)
            return x

        @staticmethod
        def _composite(in_channels, growth_rate, drop_rate):
            """
            Create a composite layer in Dense Block
            
            Args:
                in_channels (int): Number of input channels
                growth_rate (int): Growth rate for dense block
                drop_rate (float): Dropout rate
            
            Returns:
                tf.keras.Sequential: Composite layer
            """
            layer = [
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(
                    filters=growth_rate, 
                    kernel_size=3, 
                    strides=1, 
                    padding='same', 
                    use_bias=False
                )
            ]
            
            if drop_rate > 0:
                layer.append(layers.Dropout(drop_rate))
            
            return tf.keras.Sequential(layer)


class _conv_block(layers.Layer):
    """
    Convolutional block for Dense UNet
    """
    def __init__(self, in_channels, layers, growth_rate, name='conv_block'):
        """
        Initialize convolutional block
        
        Args:
            in_channels (int): Number of input channels
            layers (int): Number of layers in dense block
            growth_rate (int): Growth rate for dense block
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        self.dense_block = DenseUNet._dense_block(
            in_channels,
            layers=layers,
            growth_rate=growth_rate,
            drop_rate=0
        )
        self.trans_down = DenseUNet._trans_down(
            in_channels+layers*growth_rate, 
            in_channels, 
            drop_rate=0
        )

    def call(self, x, training=False):
        """
        Forward pass through convolutional block
        
        Args:
            x (tf.Tensor): Input tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tuple: Downsampled tensor and dense block output
        """
        link = self.dense_block(x, training=training)
        return self.trans_down(link), link


class _up_block(layers.Layer):
    """
    Upsampling block for Dense UNet
    """
    def __init__(self, in_channels, link_channels, layers, growth_rate,
                 no_conv_t=False, name='up_block'):
        """
        Initialize upsampling block
        
        Args:
            in_channels (int): Number of input channels
            link_channels (int): Number of link channels
            layers (int): Number of layers in dense block
            growth_rate (int): Growth rate for dense block
            no_conv_t (bool, optional): Whether to use upsampling instead of transposed convolution
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        tu_out_channels = link_channels - layers * growth_rate
        
        self.trans_up = DenseUNet._trans_up(
            in_channels,
            tu_out_channels,
            no_conv_t
        )
        
        self.dense_block = DenseUNet._dense_block(
            tu_out_channels+link_channels,
            layers=layers,
            growth_rate=growth_rate,
            drop_rate=0
        )

    def call(self, x, link, training=False):
        """
        Forward pass through upsampling block
        
        Args:
            x (tf.Tensor): Input tensor
            link (tf.Tensor): Link tensor from skip connection
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Processed tensor
        """
        return self.dense_block(
            tf.concat([self.trans_up(x), link], axis=-1), 
            training=training
        )
