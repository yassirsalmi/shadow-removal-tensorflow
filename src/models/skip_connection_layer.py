import tensorflow as tf
import tensorflow.keras.layers as layers

from . import opt_layers


class SkipConnectionLayer(layers.Layer):
    """
    Defines a Unet submodule with skip connection.
    +--------------------identity-------------------+
    |__ downsampling __ [submodule] __ upsampling __|
    """

    def __init__(self, 
                 down_block, 
                 up_block,
                 submodule=None,
                 use_selu=False,
                 drop_rate=0.0,
                 name='skip_connection_layer'):
        """
        Construct a Unet submodule with skip connections.
        
        Args:
            down_block (callable): Downsampling block
            up_block (callable): Upsampling block
            submodule (callable, optional): Previously defined submodules
            use_selu (bool, optional): Whether to use SELU activation
            drop_rate (float, optional): Dropout rate
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        self.downsample = down_block
        self.submodule = submodule
        self.upsample = up_block
        self.dropout = opt_layers.get_dropout(
            use_selu=use_selu, 
            drop_rate=drop_rate
        )

    def call(self, x, training=False):
        """
        Forward pass through the skip connection layer
        
        Args:
            x (tf.Tensor): Input tensor
            training (bool, optional): Whether in training mode
        
        Returns:
            tf.Tensor: Output tensor
        """
        y, link = self.downsample(x)
        
        if self.submodule is not None:
            y = self.submodule(y, training=training)
        
        z = self.upsample(y, link)
        
        if self.dropout is not None:
            return self.dropout(z, training=training)
        else:
            return z
