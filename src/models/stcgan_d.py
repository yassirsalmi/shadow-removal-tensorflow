import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers


class NLayerDiscriminator(keras.Model):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_shape, ndf=64, n_layers=3, norm_layer=layers.BatchNormalization,
                 use_sigmoid=False, name='n_layer_discriminator'):
        """
        Construct a PatchGAN discriminator
        
        Args:
            input_shape (tuple): Input shape (H, W, C)
            ndf (int): Number of filters in the last conv layer
            n_layers (int): Number of conv layers in the discriminator
            norm_layer: Normalization layer
            use_sigmoid (bool): Whether to use sigmoid activation
            name (str): Layer name
        """
        super().__init__(name=name)
        
        # Get input channels from shape
        if isinstance(input_shape[-1], int):
            in_channels = input_shape[-1]
        else:
            raise ValueError("Input shape must specify number of channels")
            
        # Ensure input_shape has batch dimension
        if len(input_shape) == 3:
            input_shape = (None,) + tuple(input_shape)
            
        # First layer with explicit input shape
        inputs = layers.Input(shape=input_shape[1:])  # Remove batch dim for Input layer
        x = inputs
        
        # First layer doesn't use normalization
        x = layers.Conv2D(
            ndf, kernel_size=4, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        )(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Increase number of filters progressively
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            x = layers.Conv2D(
                ndf * nf_mult, kernel_size=4, strides=2, padding='same',
                use_bias=False,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
            )(x)
            x = norm_layer()(x)
            x = layers.LeakyReLU(0.2)(x)
            
        # One more layer with increased filters
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        x = layers.Conv2D(
            ndf * nf_mult, kernel_size=4, strides=1, padding='same',
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        )(x)
        x = norm_layer()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Final layer outputs one channel prediction map
        x = layers.Conv2D(
            1, kernel_size=4, strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        )(x)
        if use_sigmoid:
            x = layers.Activation('sigmoid')(x)
            
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=x)
        
    def call(self, inputs, training=None):
        """Forward pass"""
        # Handle the case when inputs is a list of tensors
        if isinstance(inputs, (list, tuple)):
            # Inputs should be [rgb_image, target/output, mask]
            # rgb_image: (B, H, W, 3)
            # target/output: (B, H, W, C) where C is 1 for D1 and 3 for D2
            # mask: (B, H, W, 1)
            rgb_image, target, mask = inputs
            
            # Simple concatenation without any channel adjustments
            inputs = tf.concat([rgb_image, target, mask], axis=-1)
            
        return self.model(inputs, training=training)
