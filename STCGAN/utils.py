import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def mkdir(path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        path (str): Path to directory
    """
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)
    return


def get_sp(shadowed, shadowless, ksize: int = 5, deg: int = 1):
    """
    Calculate shadow parameters based on neighboring region
    
    Args:
        shadowed (np.ndarray): Shadowed image
        shadowless (np.ndarray): Shadowless image
        ksize (int, optional): Kernel size. Defaults to 5.
        deg (int, optional): Polynomial degree. Defaults to 1.
    
    Returns:
        np.ndarray: Shadow parameters
    """
    assert shadowed.dtype == shadowless.dtype
    
    # Prevent division by zero
    shadowed[shadowed == 0] = 1
    
    # Compute shadow parameters
    sp = shadowless.astype(np.float32) / shadowed.astype(np.float32)
    return sp


def apply_sp(shadowed, sp):
    """
    Apply shadow parameters to shadowed image
    
    Args:
        shadowed (np.ndarray): Shadowed image
        sp (np.ndarray): Shadow parameters
    
    Returns:
        np.ndarray: Restored image
    """
    if shadowed.dtype == np.uint8:
        return np.clip((sp * shadowed), 0, 255).astype(np.uint8)
    else:  # np.float32
        return np.clip((sp * shadowed), 0, 1).astype(np.float32)


def uint2float(array):
    """
    Convert uint8 array to float32 in range [0, 1]
    
    Args:
        array (np.ndarray): Input uint8 array
    
    Returns:
        np.ndarray: Float32 array
    """
    assert array.dtype == np.uint8
    return array.astype(np.float32) / 255


def float2uint(array):
    """
    Convert float32 array to uint8 in range [0, 255]
    
    Args:
        array (np.ndarray): Input float array
    
    Returns:
        np.ndarray: Uint8 array
    """
    assert (array.dtype == np.float32) or (array.dtype == np.float64)
    return (array * 255).astype(np.uint8)


def normalize_ndarray(array):
    """
    Normalize array using percentile-based scaling
    
    Args:
        array (np.ndarray): Input array
    
    Returns:
        np.ndarray: Normalized uint8 array
    """
    lower = np.percentile(array, 3)
    upper = np.percentile(array, 97)
    img = (array - lower) / (upper - lower)
    return float2uint(img)


def tf_uint2float(tensor):
    """
    Convert uint8 tensor to float32 in range [0, 1]
    
    Args:
        tensor (tf.Tensor): Input uint8 tensor
    
    Returns:
        tf.Tensor: Float32 tensor
    """
    assert tensor.dtype == tf.uint8
    return tf.cast(tensor, tf.float32) / 255


def tf_float2uint(tensor):
    """
    Convert float32 tensor to uint8 in range [0, 255]
    
    Args:
        tensor (tf.Tensor): Input float tensor
    
    Returns:
        tf.Tensor: Uint8 tensor
    """
    assert tensor.dtype in [tf.float32, tf.float64]
    return tf.cast(tensor * 255, tf.uint8)


def tf_normalize_tensor(tensor):
    """
    Normalize tensor using percentile-based scaling
    
    Args:
        tensor (tf.Tensor): Input tensor
    
    Returns:
        tf.Tensor: Normalized uint8 tensor
    """
    lower = tfp.stats.percentile(tensor, 3)
    upper = tfp.stats.percentile(tensor, 97)
    img = (tensor - lower) / (upper - lower)
    return tf_float2uint(img)
