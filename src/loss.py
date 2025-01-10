"""
Custom loss functions for shadow removal
"""

import tensorflow as tf
import tensorflow.keras.applications.vgg19 as vgg19
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models


class DataLoss(layers.Layer):
    """
    Loss between shadow parameters
    
    Supports different norm types and reduction methods
    """
    def __init__(self, 
                 norm='l1', 
                 reduction=losses.Reduction.AUTO, 
                 name='data_loss'):
        """
        Initialize DataLoss
        
        Args:
            norm (str, optional): Norm type. Defaults to 'l1'.
            reduction (tf.keras.losses.Reduction, optional): Reduction method.
            name (str, optional): Layer name.
        """
        super().__init__(name=name)
        self.norm = norm
        self.reduction = reduction

    def call(self, y_pred, y_target):
        """
        Compute loss between prediction and target
        
        Args:
            y_pred (tf.Tensor): Predicted tensor
            y_target (tf.Tensor): Target tensor
        
        Returns:
            tf.Tensor: Computed loss
        """
        if self.norm == 'l1':
            return tf.reduce_mean(tf.abs(y_pred - y_target), reduction=self.reduction)
        elif self.norm == 'l2':
            return tf.reduce_mean(tf.square(y_pred - y_target), reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm}")


class VisualLoss(layers.Layer):
    """
    Feature reconstruction perceptual loss with VGG-19.
    Measured by the norms between the features after passing
    through the pool4 layer.
    """
    def __init__(self, 
                 norm='mse', 
                 reduction=losses.Reduction.MEAN,
                 name='visual_loss'):
        super().__init__(name=name)
        self.norm = norm
        self.reduction = reduction
        
        base_model = vgg19.VGG19(weights='imagenet', include_top=False)
        layer_name = 'block4_pool'  
        outputs = base_model.get_layer(layer_name).output
        self.vgg_model = models.Model(inputs=base_model.input, outputs=outputs)
        self.vgg_model.trainable = False
        
        self.mean = tf.constant([0.485, 0.456, 0.406])
        self.std = tf.constant([0.229, 0.224, 0.225])

    def normalize_vgg(self, x):
        """Normalize images for VGG processing"""
        x = x * 0.5 + 0.5
        x = (x - self.mean) / self.std
        return x

    def call(self, y_pred, y_target):
        """Compute perceptual loss"""
        y_pred = self.normalize_vgg(y_pred)
        y_target = self.normalize_vgg(y_target)
        
        feat_pred = self.vgg_model(y_pred)
        feat_target = self.vgg_model(y_target)
        
        if self.norm == 'mse':
            return tf.reduce_mean(tf.square(feat_pred - feat_target))
        else:  # l1
            return tf.reduce_mean(tf.abs(feat_pred - feat_target))


class AdversarialLoss(layers.Layer):
    """
    Objective of a conditional GAN:
    E_(x,y){[log(D(x, y)]} + E_(x,z){[log(1-D(x, G(x, z))}
    """
    def __init__(self, 
                 ls=False,  
                 rel=False, 
                 avg=False, 
                 name='adversarial_loss'):
        super().__init__(name=name)
        self.real_label = tf.constant(1.0)
        self.fake_label = tf.constant(-1.0 if ls else 0.0)
        self.ls = ls
        self.rel = rel
        self.avg = avg

    def cal_loss(self, c_out, label):
        """Calculate loss based on type"""
        label = tf.broadcast_to(label, c_out.shape)
        if not self.ls:
            return tf.reduce_mean(tf.square(c_out - label))
        else:
            return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=label, logits=c_out))

    def call(self, c_real, c_fake, d_loss=True):
        """
        Compute adversarial loss
        
        Args:
            c_real: Discriminator output for real images
            c_fake: Discriminator output for fake images
            d_loss: If True, compute discriminator loss, else generator loss
        """
        if d_loss:
            if self.rel:
                if self.avg:  # RaGAN
                    loss_real = self.cal_loss(
                        c_real - tf.reduce_mean(c_fake, axis=0),
                        self.real_label)
                    loss_fake = self.cal_loss(
                        c_fake - tf.reduce_mean(c_real, axis=0),
                        self.fake_label)
                    return (loss_real + loss_fake) * 0.5
                else:  # RpGAN
                    return self.cal_loss(c_real - c_fake, self.real_label)
            else:  # SGAN
                loss_real = self.cal_loss(c_real, self.real_label)
                loss_fake = self.cal_loss(c_fake, self.fake_label)
                return (loss_real + loss_fake) * 0.5
        else:  # Generator loss
            if self.rel:
                if self.avg:  # RaGAN
                    loss_real = self.cal_loss(
                        c_real - tf.reduce_mean(c_fake, axis=0),
                        self.fake_label)
                    loss_fake = self.cal_loss(
                        c_fake - tf.reduce_mean(c_real, axis=0),
                        self.real_label)
                    return (loss_real + loss_fake) * 0.5
                else:  # RpGAN
                    return self.cal_loss(c_fake - c_real, self.real_label)
            else:  # SGAN
                return self.cal_loss(c_fake, self.real_label)


class SoftAdapt(layers.Layer):
    """
    Adaptive multi-loss weighting strategy
    
    Dynamically adjusts loss weights during training
    """
    def __init__(self, 
                 losses, 
                 init_weights=None, 
                 beta=0.1, 
                 epsilon=1e-8, 
                 min_=1e-4, 
                 weighted=True, 
                 normalized=True,
                 name='soft_adapt'):
        """
        Initialize SoftAdapt
        
        Args:
            losses (list): List of loss functions
            init_weights (list, optional): Initial loss weights
            beta (float, optional): Learning rate for weight update
            epsilon (float, optional): Small constant to prevent division by zero
            min_ (float, optional): Minimum weight value
            weighted (bool, optional): Whether to use weighted losses
            normalized (bool, optional): Whether to normalize weights
            name (str, optional): Layer name
        """
        super().__init__(name=name)
        self.losses = losses
        self.size = len(losses)
        
        if init_weights is None:
            self.weights = tf.ones(self.size) / self.size
        else:
            assert len(init_weights) == self.size
            self.weights = tf.convert_to_tensor(init_weights, dtype=tf.float32)
            self.weights /= tf.reduce_sum(self.weights)
        
        self.current_loss = tf.ones(self.size)
        self.prev_loss = tf.ones(self.size)
        self.gradient = tf.zeros(self.size)
        
        self.beta = beta
        self.epsilon = epsilon
        self.weighted = weighted
        self.normalized = normalized
        self.alpha = 0.9 
        self.min_ = min_

    def update(self, losses):
        """
        Update current loss values
        
        Args:
            losses (dict): Dictionary of loss values
        """
        self.prev_loss = self.current_loss
        self.current_loss = tf.convert_to_tensor(list(losses.values()))

    def update_weights(self):
        """
        Adaptive weight update strategy
        """
        self.gradient = (self.current_loss - self.prev_loss) / (
            tf.abs(self.prev_loss) + self.epsilon
        )
        
        delta = self.beta * self.gradient
        self.weights = tf.maximum(
            self.weights * tf.exp(delta), 
            self.min_
        )
        
        if self.normalized:
            self.weights /= tf.reduce_sum(self.weights)

    def call(self, losses, update_weights=False):
        """
        Compute weighted loss
        
        Args:
            losses (dict): Dictionary of loss values
            update_weights (bool, optional): Whether to update weights
        
        Returns:
            tf.Tensor: Weighted total loss
        """
        self.update(losses)
        
        if update_weights:
            self.update_weights()
        
        if self.weighted:
            total_loss = tf.reduce_sum(
                [loss * weight for loss, weight in zip(losses.values(), self.weights)]
            )
        else:
            total_loss = tf.reduce_sum(list(losses.values()))
        
        return total_loss

    def get_weights(self):
        """
        Get current loss weights
        
        Returns:
            tf.Tensor: Current loss weights
        """
        return self.weights
