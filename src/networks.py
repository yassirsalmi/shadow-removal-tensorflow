from enum import Enum, unique

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers

from src.models.began import BEGAN
from src.models.denseunet import DenseUNet
from src.models.mnet import MNet
from src.models.patchgan import PatchGAN
from src.models.stcgan_d import NLayerDiscriminator
from src.models.stcgan_g import UnetGenerator
from src.models.unet import UNet
from src.models.dummy import DummyNet


def weights_init(model):
    """
    Custom weights initialization for network models
    Matches PyTorch's default initialization
    """
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.Conv2DTranspose)):
            if isinstance(layer.kernel_initializer, initializers.TruncatedNormal):
                continue
                
            if hasattr(layer, 'input_shape') and layer.input_shape is not None and None not in layer.input_shape:
                fan_in = layer.kernel_size[0] * layer.kernel_size[1] * layer.input_shape[-1]
                std = 1. / tf.sqrt(tf.cast(fan_in, tf.float32))
                layer.kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=std)
                if layer.bias is not None:
                    layer.bias_initializer = initializers.Zeros()
                
        elif isinstance(layer, layers.BatchNormalization):
            layer.gamma_initializer = initializers.Ones()
            layer.beta_initializer = initializers.Zeros()
            layer.moving_mean_initializer = initializers.Zeros()
            layer.moving_variance_initializer = initializers.Ones()
            
        elif isinstance(layer, layers.Dense):
            if isinstance(layer.kernel_initializer, initializers.TruncatedNormal):
                continue
                
            if hasattr(layer, 'input_shape') and layer.input_shape is not None and None not in layer.input_shape:
                fan_in = layer.input_shape[-1]
                std = 1. / tf.sqrt(tf.cast(fan_in, tf.float32))
                layer.kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=std)
                if layer.bias is not None:
                    layer.bias_initializer = initializers.Zeros()
                
        elif hasattr(layer, 'layers'):
            weights_init(layer)  # Recursively initialize custom layers
    
    return model


class BaseNetwork(keras.Model):
    """Base network class with PyTorch-like initialization"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.built = False
        
    def build(self, input_shape):
        """Build model and initialize weights"""
        super().build(input_shape)
        if not self.built:
            self.built = True
            weights_init(self)


@unique
class Generators(Enum):
    """Available generator architectures"""
    UNET = UNet
    MNET = MNet
    DENSEUNET = DenseUNet
    STCGAN = UnetGenerator


@unique
class Discriminators(Enum):
    """Available discriminator architectures"""
    PATCHGAN = PatchGAN
    BEGAN = BEGAN
    STCGAN = NLayerDiscriminator
    DUMMY = DummyNet


def get_generator(key: str, *args, **kwargs):
    """
    Get generator model by key
    
    Args:
        key (str): Generator architecture name
        *args: Arguments for generator constructor
        **kwargs: Keyword arguments for generator constructor
        
    Returns:
        keras.Model: Initialized generator model
    """
    try:
        generator_class = Generators[key.upper()].value
        model = generator_class(*args, **kwargs)
        if not isinstance(model, BaseNetwork):
            model = weights_init(model)
        return model
    except KeyError:
        raise ValueError(f"Unknown generator architecture: {key}")


def get_discriminator(key: str, *args, **kwargs):
    """
    Get discriminator model by key
    
    Args:
        key (str): Discriminator architecture name
        *args: Arguments for discriminator constructor
        **kwargs: Keyword arguments for discriminator constructor
        
    Returns:
        keras.Model: Initialized discriminator model
    """
    try:
        discriminator_class = Discriminators[key.upper()].value
        model = discriminator_class(*args, **kwargs)
        if not isinstance(model, BaseNetwork):
            model = weights_init(model)
        return model
    except KeyError:
        raise ValueError(f"Unknown discriminator architecture: {key}")
