import numpy as np
import tensorflow as tf


def color_adjustment(shadow_free, shadow, shadow_mask):
    """
    Perform color adjustment using linear regression
    
    Args:
        shadow_free (np.ndarray): Shadow-free image
        shadow (np.ndarray): Shadowed image
        shadow_mask (np.ndarray): Binary shadow mask
    
    Returns:
        tuple: Corrected image and regression parameters
    """
    # Ensure inputs are numpy arrays
    shadow_free = np.array(shadow_free)
    shadow = np.array(shadow)
    shadow_mask = np.array(shadow_mask)
    
    # Replicate shadow mask across color channels
    shadow_mask = np.repeat(shadow_mask[..., np.newaxis], 3, axis=-1)
    
    # Select non-shadow pixels
    source = shadow_free[shadow_mask == 0].astype(np.float32) / 255.0
    target = shadow[shadow_mask == 0].astype(np.float32) / 255.0
    
    # Reshape to separate color channels
    source = source.reshape(-1, 3)
    target = target.reshape(-1, 3)
    
    # Perform linear regression for each color channel
    linear_params = []
    for channel in range(3):
        # Add bias term
        X = np.column_stack([np.ones_like(source[:, channel]), source[:, channel]])
        
        # Linear regression using numpy's least squares
        params = np.linalg.lstsq(X, target[:, channel], rcond=None)[0]
        linear_params.append(params)
    
    # Flatten parameters
    param = np.concatenate(linear_params)
    
    # Apply color correction
    corrected_im = shadow_free.astype(np.float32) / 255.0
    for channel in range(3):
        corrected_im[..., channel] = (
            corrected_im[..., channel] * param[channel*2 + 1] + param[channel*2]
        )
    
    # Convert back to uint8
    corrected_im = np.clip(corrected_im * 255, 0, 255).astype(np.uint8)
    
    return corrected_im, param


def tf_color_adjustment(shadow_free, shadow, shadow_mask):
    """
    TensorFlow implementation of color adjustment
    
    Args:
        shadow_free (tf.Tensor): Shadow-free image
        shadow (tf.Tensor): Shadowed image
        shadow_mask (tf.Tensor): Binary shadow mask
    
    Returns:
        tuple: Corrected image and regression parameters
    """
    # Convert to float32 tensors
    shadow_free = tf.cast(shadow_free, tf.float32) / 255.0
    shadow = tf.cast(shadow, tf.float32) / 255.0
    shadow_mask = tf.cast(shadow_mask, tf.float32)
    
    # Replicate shadow mask across color channels
    shadow_mask = tf.repeat(shadow_mask[..., tf.newaxis], 3, axis=-1)
    
    # Select non-shadow pixels
    source = tf.boolean_mask(shadow_free, shadow_mask == 0)
    target = tf.boolean_mask(shadow, shadow_mask == 0)
    
    # Reshape to separate color channels
    source = tf.reshape(source, [-1, 3])
    target = tf.reshape(target, [-1, 3])
    
    # Perform linear regression for each color channel
    linear_params = []
    for channel in range(3):
        # Add bias term
        X = tf.concat([
            tf.ones((tf.shape(source)[0], 1), dtype=tf.float32),
            source[:, channel:channel+1]
        ], axis=1)
        
        # Linear regression using TensorFlow
        params = tf.linalg.lstsq(X, target[:, channel:channel+1])
        linear_params.append(tf.squeeze(params))
    
    # Flatten parameters
    param = tf.concat(linear_params, axis=0)
    
    # Apply color correction
    corrected_im = shadow_free
    for channel in range(3):
        corrected_im = tf.tensor_scatter_nd_update(
            corrected_im, 
            tf.where(shadow_mask[..., channel] == 0),
            corrected_im[..., channel] * param[channel*2 + 1] + param[channel*2]
        )
    
    # Convert back to uint8
    corrected_im = tf.clip_by_value(corrected_im * 255, 0, 255)
    corrected_im = tf.cast(corrected_im, tf.uint8)
    
    return corrected_im, param


# Expose both NumPy and TensorFlow versions
__all__ = ['color_adjustment', 'tf_color_adjustment']
