import os
import numpy as np
import tensorflow as tf
from tensorflow_probability import stats as tfp


def mkdir(path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        path (str): Path to directory
    """
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)
    return


def get_sp(shadowed: np.ndarray, 
           shadowless: np.ndarray, 
           ksize: int = 5, 
           deg: int = 1) -> np.ndarray:
    """
    Calculate shadow parameters based on neighboring region
    
    Args:
        shadowed (np.ndarray): Image with shadows
        shadowless (np.ndarray): Shadow-free image
        ksize (int, optional): Kernel size. Defaults to 5.
        deg (int, optional): Polynomial degree. Defaults to 1.
    
    Returns:
        np.ndarray: Shadow parameters
    """
    assert shadowed.dtype == shadowless.dtype
    
    shadowed = shadowed.astype(np.float32)
    shadowless = shadowless.astype(np.float32)
    
    shadowed = np.where(shadowed == 0, 1e-8, shadowed)
    
    sp = shadowless / shadowed
    return sp


def apply_sp(shadowed: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """
    Apply shadow parameters to shadowed image
    
    Args:
        shadowed (np.ndarray): Shadowed image
        sp (np.ndarray): Shadow parameters
    
    Returns:
        np.ndarray: Shadow-corrected image
    """
    if shadowed.dtype == np.uint8:
        return np.clip((sp * shadowed), 0, 255).astype(np.uint8)
    else:  # np.float32
        return np.clip((sp * shadowed), 0, 1).astype(np.float32)


def uint2float(array: np.ndarray) -> np.ndarray:
    """
    Convert uint8 image to float32
    
    Args:
        array (np.ndarray): Uint8 image
    
    Returns:
        np.ndarray: Float32 image
    """
    assert array.dtype == np.uint8
    return array.astype(np.float32) / 255.0


def float2uint(array: np.ndarray) -> np.ndarray:
    """
    Convert float image to uint8
    
    Args:
        array (np.ndarray): Float image
    
    Returns:
        np.ndarray: Uint8 image
    """
    assert (array.dtype == np.float32) or (array.dtype == np.float64)
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)


def normalize_ndarray(array: np.ndarray, 
                      lower_percentile: float = 3.0, 
                      upper_percentile: float = 97.0) -> np.ndarray:
    """
    Normalize array using percentile-based method
    
    Args:
        array (np.ndarray): Input array
        lower_percentile (float, optional): Lower percentile. Defaults to 3.0.
        upper_percentile (float, optional): Upper percentile. Defaults to 97.0.
    
    Returns:
        np.ndarray: Normalized uint8 array
    """
    lower = np.percentile(array, lower_percentile)
    upper = np.percentile(array, upper_percentile)
    
    if upper == lower:
        return float2uint(np.zeros_like(array))
    
    img = (array - lower) / (upper - lower)
    return float2uint(img)


def tf_uint2float(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert uint8 tensor to float32
    
    Args:
        tensor (tf.Tensor): Uint8 tensor
    
    Returns:
        tf.Tensor: Float32 tensor
    """
    return tf.cast(tensor, tf.float32) / 255.0


def tf_float2uint(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert float tensor to uint8
    
    Args:
        tensor (tf.Tensor): Float tensor
    
    Returns:
        tf.Tensor: Uint8 tensor
    """
    return tf.cast(tf.clip_by_value(tensor, 0, 1) * 255, tf.uint8)


def tf_normalize_tensor(tensor: tf.Tensor, 
                        lower_percentile: float = 3.0, 
                        upper_percentile: float = 97.0) -> tf.Tensor:
    """
    Normalize tensor using percentile-based method
    
    Args:
        tensor (tf.Tensor): Input tensor
        lower_percentile (float, optional): Lower percentile. Defaults to 3.0.
        upper_percentile (float, optional): Upper percentile. Defaults to 97.0.
    
    Returns:
        tf.Tensor: Normalized uint8 tensor
    """
    lower = tfp.stats.percentile(tensor, lower_percentile)
    upper = tfp.stats.percentile(tensor, upper_percentile)
    
    tensor = tf.where(
        tf.math.equal(upper, lower), 
        tf.zeros_like(tensor), 
        (tensor - lower) / (upper - lower)
    )
    
    return tf_float2uint(tensor)
