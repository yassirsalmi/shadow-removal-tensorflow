import tensorflow as tf
import tensorflow.keras.layers as layers


class DummyNet(layers.Layer):
    """
    Dummy neural network for testing or placeholder purposes
    
    Simply performs a 1x1 convolution to change number of channels
    """

    def __init__(self, 
                 in_channels, 
                 out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=True,
                 use_selu=False,
                 activation=None,
                 name='dummy_net'):
        """
        Initialize DummyNet
        
        Args:
            in_channels (int): Number of input image channels
            out_channels (int): Number of output image channels
            ngf (int, optional): Base number of filters (unused)
            drop_rate (float, optional): Dropout rate (unused)
            no_conv_t (bool, optional): Transposed convolution flag (unused)
            use_selu (bool, optional): SELU activation flag (unused)
            activation (str or tf.keras.layers.Layer, optional): Activation layer (unused)
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        self.out_channels = out_channels
        self.dummy_conv = layers.Conv2D(
            filters=out_channels, 
            kernel_size=1, 
            strides=1, 
            padding='valid'
        )

    def call(self, x, training=False):
        """
        Forward pass through DummyNet
        
        Args:
            x (tf.Tensor): Input image tensor
            training (bool, optional): Whether in training mode (unused)
        
        Returns:
            tf.Tensor: Output tensor with modified channel count
        """
        return self.dummy_conv(x)
