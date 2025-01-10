#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom loss functions for shadow removal using TensorFlow
"""

import tensorflow as tf
import tensorflow.keras.applications.vgg19 as vgg19
from tensorflow.keras import layers

from tensorflow.STCGAN.dataset_h5 import ISTDDataset


class DataLoss(layers.Layer):
    """
    Loss between shadow parameters with configurable norm
    """

    def __init__(self, 
                 norm='l1', 
                 reduction='mean', 
                 name='data_loss', 
                 **kwargs):
        """
        Initialize DataLoss layer
        
        Args:
            norm (str, optional): Norm type. Defaults to 'l1'.
            reduction (str, optional): Reduction method. Defaults to 'mean'.
        """
        super().__init__(name=name, **kwargs)
        self.reduction = reduction
        self.norm = norm

    def call(self, y_pred, y_target):
        """
        Compute data loss
        
        Args:
            y_pred (tf.Tensor): Predicted values
            y_target (tf.Tensor): Target values
        
        Returns:
            tf.Tensor: Computed loss
        """
        if self.norm == 'l1':
            return tf.reduce_mean(tf.abs(y_pred - y_target))
        elif self.norm == 'l2':
            return tf.reduce_mean(tf.square(y_pred - y_target))
        else:
            raise ValueError(f"Unsupported norm type: {self.norm}")


class VisualLoss(layers.Layer):
    """
    Feature reconstruction perceptual loss using VGG-19
    Measures feature differences after pool4 layer
    """

    def __init__(self, 
                 norm='mse', 
                 reduction='mean', 
                 name='visual_loss', 
                 **kwargs):
        """
        Initialize VisualLoss layer
        
        Args:
            norm (str, optional): Norm type. Defaults to 'mse'.
            reduction (str, optional): Reduction method. Defaults to 'mean'.
        """
        super().__init__(name=name, **kwargs)
        self.reduction = reduction
        self.norm = norm
        
        # Load pre-trained VGG19 and extract features
        vgg_model = vgg19.VGG19(include_top=False, weights='imagenet')
        self.vgg_features = tf.keras.Model(
            inputs=vgg_model.input, 
            outputs=vgg_model.get_layer('block4_pool').output
        )
        
        # Freeze VGG layers
        self.vgg_features.trainable = False
        
        # Dataset normalization parameters
        self.mean = tf.constant(ISTDDataset.mean, dtype=tf.float32)
        self.std = tf.constant(ISTDDataset.std, dtype=tf.float32)

    def call(self, x, y_pred, img_target):
        """
        Compute visual loss
        
        Args:
            x (tf.Tensor): Input image
            y_pred (tf.Tensor): Predicted image
            img_target (tf.Tensor): Target image
        
        Returns:
            tf.Tensor: Computed visual loss
        """
        # Denormalize input
        img_in = x * self.std + self.mean
        
        # Predict image
        img_pred = tf.clip_by_value(y_pred * img_in, 0, 1)
        
        # Extract features
        feature_pred = self.vgg_features(img_pred)
        feature_target = self.vgg_features(img_target)
        
        # Compute loss
        if self.norm == 'mse':
            return tf.reduce_mean(tf.square(feature_pred - feature_target))
        elif self.norm == 'mae':
            return tf.reduce_mean(tf.abs(feature_pred - feature_target))
        else:
            raise ValueError(f"Unsupported norm type: {self.norm}")


class AdversarialLoss(layers.Layer):
    """
    Adversarial loss for conditional GAN
    Supports least squares and binary cross-entropy losses
    """

    def __init__(self, 
                 ls=False, 
                 rel=False, 
                 avg=False, 
                 name='adversarial_loss', 
                 **kwargs):
        """
        Initialize AdversarialLoss layer
        
        Args:
            ls (bool, optional): Use least squares loss. Defaults to False.
            rel (bool, optional): Use relative loss. Defaults to False.
            avg (bool, optional): Use average loss. Defaults to False.
        """
        super().__init__(name=name, **kwargs)
        self.ls = ls
        self.rel = rel
        self.avg = avg
        
        # Labels for real and fake samples
        self.real_label = 1.0 if not ls else 1.0
        self.fake_label = 0.0 if not ls else -1.0

    def call(self, D_out, is_real):
        """
        Compute adversarial loss
        
        Args:
            D_out (tf.Tensor): Discriminator output
            is_real (bool): Whether the sample is real
        
        Returns:
            tf.Tensor: Computed adversarial loss
        """
        # Create target tensor
        target = tf.ones_like(D_out) * (self.real_label if is_real else self.fake_label)
        
        # Compute loss
        if not self.ls:
            return tf.keras.losses.binary_crossentropy(target, D_out, from_logits=True)
        else:
            return tf.keras.losses.mean_squared_error(target, D_out)


class SoftAdapt(layers.Layer):
    """
    Adaptive loss weighting mechanism
    Dynamically adjusts loss weights during training
    """

    def __init__(self, 
                 losses: list, 
                 init_weights=None, 
                 beta=0.1, 
                 epsilon=1e-8, 
                 min_=1e-4, 
                 weighted=True, 
                 normalized=True, 
                 name='soft_adapt', 
                 **kwargs):
        """
        Initialize SoftAdapt layer
        
        Args:
            losses (list): List of loss names
            init_weights (list, optional): Initial loss weights
            beta (float, optional): Adaptation rate. Defaults to 0.1.
            epsilon (float, optional): Numerical stability term. Defaults to 1e-8.
            min_ (float, optional): Minimum weight. Defaults to 1e-4.
            weighted (bool, optional): Use weighted adaptation. Defaults to True.
            normalized (bool, optional): Normalize gradients. Defaults to True.
        """
        super().__init__(name=name, **kwargs)
        
        self.loss_names = losses
        self.size = len(losses)
        
        # Initialize weights
        if init_weights is None:
            init_weights = tf.ones(self.size) / self.size
        else:
            init_weights = tf.convert_to_tensor(init_weights, dtype=tf.float32)
            init_weights /= tf.reduce_sum(init_weights)
        
        # Trainable variables
        self.weights = tf.Variable(init_weights, trainable=False)
        
        # Adaptive parameters
        self.current_loss = tf.Variable(tf.ones(self.size), trainable=False)
        self.prev_loss = tf.Variable(tf.ones(self.size), trainable=False)
        self.gradient = tf.Variable(tf.zeros(self.size), trainable=False)
        
        # Hyperparameters
        self.beta = beta
        self.epsilon = epsilon
        self.weighted = weighted
        self.normalized = normalized
        self.alpha = 0.9  # smoothing factor
        self.min_ = min_

    def update(self, losses: dict):
        """
        Update current losses
        
        Args:
            losses (dict): Dictionary of losses
        """
        loss_list = [losses[k] for k in self.loss_names]
        self.current_loss.assign(tf.stack(loss_list))

    def update_weights(self):
        """
        Dynamically update loss weights
        """
        # Compute gradient and update
        loss_detached = tf.stop_gradient(self.current_loss)
        self.gradient.assign(loss_detached - self.prev_loss)
        
        # Normalize gradient if specified
        grad = self.gradient
        if self.normalized:
            grad /= tf.maximum(self.prev_loss, self.epsilon)
        
        # Adjust gradient
        grad -= tf.reduce_max(grad)
        
        # Compute new weights
        new_weight = tf.nn.softmax(self.beta * grad)
        
        # Apply weighted adaptation
        if self.weighted:
            new_weight *= (tf.reduce_sum(self.prev_loss) - self.prev_loss)
            new_weight /= tf.reduce_sum(new_weight)
        
        # Update weights with smoothing
        self.weights.assign(
            self.alpha * self.weights + (1 - self.alpha) * new_weight
        )
        
        # Update previous loss
        self.prev_loss.assign(loss_detached)

    def call(self, losses, update_weights=False):
        """
        Compute weighted loss
        
        Args:
            losses (dict): Dictionary of losses
            update_weights (bool, optional): Whether to update weights
        
        Returns:
            tf.Tensor: Weighted loss
        """
        self.update(losses)
        if update_weights:
            self.update_weights()
        
        return tf.reduce_sum(self.current_loss * self.weights)

    def get_loss(self):
        """
        Get current losses
        
        Returns:
            dict: Current losses
        """
        return dict(zip(self.loss_names, self.current_loss.numpy().tolist()))

    def get_weights(self):
        """
        Get current loss weights
        
        Returns:
            dict: Current loss weights
        """
        return dict(zip(self.loss_names, self.weights.numpy().tolist()))
