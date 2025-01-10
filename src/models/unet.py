"""
UNet Architecture for Image Segmentation and Translation
Inspired by:
https://github.com/mateuszbuda/brain-segmentation-pytorch
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas
    with shape features automatically extracted by a deep learning algorithm},
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

from . import opt_layers
from .skip_connection_layer import SkipConnectionLayer


class UNet(layers.Layer):
    """
    UNet architecture with flexible configuration
    """

    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=False,
                 use_selu=False,
                 activation=None,
                 name='unet'):
        """
        Initialize UNet
        
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

        # Initial bottleneck block
        block = conv(ngf*(2**(depth-1)), ngf*(2**depth), use_selu)

        # Build intermediate blocks
        for i in reversed(range(1, depth)):
            block = SkipConnectionLayer(
                _conv_block(ngf*(2**(i-1)), ngf*2**i, use_selu),
                _up_block(ngf*2**(i+1), ngf*2**i, use_selu, no_conv_t),
                submodule=block, 
                drop_rate=drop_rate
            )

        # Final block
        block = SkipConnectionLayer(
            _conv_block(in_channels, ngf, use_selu),
            _up_block(ngf*2, ngf, use_selu, no_conv_t),
            submodule=block, 
            drop_rate=0
        )

        # Build final sequence
        sequence = [
            block,
            layers.Conv2D(
                filters=out_channels, 
                kernel_size=1, 
                strides=1, 
                use_bias=False
            )
        ]
        
        # Optional activation
        if activation is not None and activation != "none":
            sequence.append(opt_layers.get_activation(activation))

        self.model = tf.keras.Sequential(sequence)

    def call(self, x, training=False):
        """
        Forward pass through UNet
        
        Args:
            x (tf.Tensor): Input image tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Processed image tensor
        """
        return self.model(x, training=training)


def conv(in_channels, features, use_selu: bool):
    """
    Create a convolutional block with optional SELU normalization
    
    Args:
        in_channels (int): Number of input channels
        features (int): Number of output features
        use_selu (bool): Whether to use SELU activation
    
    Returns:
        tf.keras.Sequential: Convolutional block
    """
    return tf.keras.Sequential([
        layers.Conv2D(
            filters=features, 
            kernel_size=3, 
            strides=1, 
            padding='same', 
            use_bias=False
        ),
        opt_layers.get_norm(use_selu, features),
        layers.Conv2D(
            filters=features, 
            kernel_size=3, 
            strides=1, 
            padding='same', 
            use_bias=False
        ),
        opt_layers.get_norm(use_selu, features)
    ])


class _conv_block(layers.Layer):
    """
    Convolutional block for UNet
    """
    def __init__(self, in_channels, features, selu, name='conv_block'):
        """
        Initialize convolutional block
        
        Args:
            in_channels (int): Number of input channels
            features (int): Number of output features
            selu (bool): Whether to use SELU activation
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        self.block = conv(in_channels, features, selu)
        self.pool = layers.MaxPool2D(pool_size=2, strides=2)

    def call(self, x, training=False):
        """
        Forward pass through convolutional block
        
        Args:
            x (tf.Tensor): Input tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tuple: Pooled output and original output
        """
        out = self.block(x, training=training)
        return self.pool(out), out


class _up_block(layers.Layer):
    """
    Upsampling block for UNet
    """
    def __init__(self, 
                 in_channels, 
                 features, 
                 selu, 
                 no_conv_t, 
                 name='up_block'):
        """
        Initialize upsampling block
        
        Args:
            in_channels (int): Number of input channels
            features (int): Number of output features
            selu (bool): Whether to use SELU activation
            no_conv_t (bool): Whether to use upsampling instead of transposed convolution
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        self.up_conv = opt_layers.get_upsample(no_conv_t, in_channels, features)
        self.conv_block = conv(2*features, features, selu)

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
        x = self.up_conv(x)
        return self.conv_block(
            tf.concat([x, link], axis=-1), 
            training=training
        )
