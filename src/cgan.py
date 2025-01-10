import datetime
import logging
import os
import time
from tqdm import tqdm

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses

import src.networks as networks
import src.transform as transform
import src.utils as utils
from src.dataset import ISTDDataset


class CGAN:
    def __init__(self, args):
        """Initialize the CGAN model"""
        self.logger = logging.getLogger(__name__)
        
        self.args = args
        
        if tf.config.list_physical_devices('GPU'):
            self.strategy = tf.distribute.MirroredStrategy()
            self.device = '/gpu:0'
        else:
            self.strategy = None
            self.device = '/cpu:0'
        self.logger.info(f"Using device: {self.device}")
        
        with tf.device(self.device):
            self._create_models(args)
    
    def _create_models(self, args):
        """Create all network models and optimizers"""
        self.logger.info("Creating network models")
        
        self.G1 = networks.get_generator(
            args.net_G,
            input_shape=(None, None, 4),
            out_channels=1,
            ngf=args.ngf,
            use_dropout=args.droprate > 0
        )
        
        self.G2 = networks.get_generator(
            args.net_G,
            input_shape=(None, None, 4),
            out_channels=3,  
            ngf=args.ngf,
            use_dropout=args.droprate > 0
        )
        
        self.D1 = networks.get_discriminator(
            args.net_D,
            input_shape=(None, None, 5),
            ndf=args.ndf,
            use_sigmoid=args.gan_mode == 'vanilla'
        )
        
        self.D2 = networks.get_discriminator(
            args.net_D,
            input_shape=(None, None, 7), 
            ndf=args.ndf,
            use_sigmoid=args.gan_mode == 'vanilla'
        )
        
        self.g1_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr_G,
            beta_1=args.beta1,
            beta_2=args.beta2,
            epsilon=1e-8
        )
        
        self.g2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr_G,
            beta_1=args.beta1,
            beta_2=args.beta2,
            epsilon=1e-8
        )
        
        self.d1_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr_D,
            beta_1=args.beta1,
            beta_2=args.beta2,
            epsilon=1e-8
        )
        
        self.d2_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr_D,
            beta_1=args.beta1,
            beta_2=args.beta2,
            epsilon=1e-8
        )
        
        self.lr_schedule_G = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.lr_G,
            decay_steps=1,
            decay_rate=1-args.decay,
            staircase=False
        )
        
        self.lr_schedule_D = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.lr_D,
            decay_steps=1,
            decay_rate=1-args.decay,
            staircase=False
        )
        
        self.adversarial_loss = self._adversarial_loss
        self.data_loss = self._data_loss
        self.visual_loss = self._visual_loss
        
        if "infer" in args.tasks and "train" not in args.tasks:
            assert args.load_weights_g1 is not None
            assert args.load_weights_g2 is not None
            self.init_weight(
                g1_weights=args.load_weights_g1,
                g2_weights=args.load_weights_g2,
                d1_weights=args.load_weights_d1,
                d2_weights=args.load_weights_d2
            )
    
    def init_weight(self, g1_weights=None, g2_weights=None, d1_weights=None, d2_weights=None):
        """Initialize model weights"""
        if g1_weights:
            self.G1.load_weights(g1_weights)
        if g2_weights:
            self.G2.load_weights(g2_weights)
        if d1_weights:
            self.D1.load_weights(d1_weights)
        if d2_weights:
            self.D2.load_weights(d2_weights)
    
    def _adversarial_loss(self, real_output, fake_output):
        """Adversarial loss using binary cross-entropy"""
        real_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(real_output), real_output, from_logits=True
        )
        fake_loss = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(fake_output), fake_output, from_logits=True
        )
        return real_loss + fake_loss
    
    def _data_loss(self, real, generated):
        """L1 data loss"""
        return tf.reduce_mean(tf.abs(real - generated))
    
    def _visual_loss(self, real, generated):
        """Perceptual loss using VGG features"""
        # Load and configure VGG19 model
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        feature_extractor = tf.keras.Model(
            inputs=vgg.input,
            outputs=vgg.get_layer('block4_conv4').output
        )
        
        real = tf.keras.applications.vgg19.preprocess_input(real * 255.0)
        generated = tf.keras.applications.vgg19.preprocess_input(generated * 255.0)
        
        real_features = feature_extractor(real)
        generated_features = feature_extractor(generated)
        
        return tf.reduce_mean(tf.square(real_features - generated_features))
    
    def _generator_loss(self, d1_fake, d2_fake, fake_masks, fake_images, shadow_masks, real_images):
        """Calculate generator loss"""
        g1_loss = tf.reduce_mean(tf.square(d1_fake - 1.0))  # LSGAN loss
        g2_loss = tf.reduce_mean(tf.square(d2_fake - 1.0))  # LSGAN loss
        
        mask_loss = tf.reduce_mean(tf.abs(shadow_masks - fake_masks))
        image_loss = tf.reduce_mean(tf.abs(real_images - fake_images))
        
        perceptual_loss = self.visual_loss(real_images, fake_images)
        
        total_loss = (g1_loss + g2_loss) * 0.5 + \
                    (mask_loss + image_loss) * 100.0 + \
                    perceptual_loss * 10.0
                    
        return total_loss
        
    def _discriminator_loss(self, d1_real, d1_fake, d2_real, d2_fake):
        """Calculate discriminator loss"""
        d1_real_loss = tf.reduce_mean(tf.square(d1_real - 1.0))
        d1_fake_loss = tf.reduce_mean(tf.square(d1_fake))
        
        d2_real_loss = tf.reduce_mean(tf.square(d2_real - 1.0))
        d2_fake_loss = tf.reduce_mean(tf.square(d2_fake))
        
        total_loss = (d1_real_loss + d1_fake_loss) * 0.5 + \
                    (d2_real_loss + d2_fake_loss) * 0.5
                    
        return total_loss
    
    def gan_loss(self, outputs, is_real, is_discriminator):
        if is_discriminator:
            if is_real:
                return tf.reduce_mean(tf.square(outputs - 1.0))
            else:
                return tf.reduce_mean(tf.square(outputs))
        else:
            if is_real:
                return tf.reduce_mean(tf.square(outputs - 1.0))
            else:
                return tf.reduce_mean(tf.square(outputs))
    
    def l1_loss(self, outputs, targets):
        return tf.reduce_mean(tf.abs(outputs - targets))
    
    @tf.function
    def train_step(self, batch):
        """Single training step"""
        real_img = batch['img']      
        real_mask = batch['mask']    
        real_target = batch['target']  
        
        with tf.GradientTape(persistent=True) as tape:
            fake_target1 = self.G1([real_img, real_mask], training=True)  
            fake_target2 = self.G2([real_img, fake_target1], training=True)
            
            d1_fake = self.D1([real_img, fake_target1, real_mask], training=True)
            d1_real = self.D1([real_img, real_target[:,:,:,0:1], real_mask], training=True)
            
            d2_fake = self.D2([real_img, fake_target2, real_mask], training=True)
            d2_real = self.D2([real_img, real_target, real_mask], training=True)
            
            g1_gan_loss = self.gan_loss(d1_fake, True, False)
            g2_gan_loss = self.gan_loss(d2_fake, True, False)
            
            g1_l1_loss = self.l1_loss(real_target[:,:,:,0:1], fake_target1)  
            g2_l1_loss = self.l1_loss(real_target, fake_target2)
            
            g1_total_loss = g1_gan_loss + self.args.lambda_L1 * g1_l1_loss
            g2_total_loss = g2_gan_loss + self.args.lambda_L1 * g2_l1_loss
            
            d1_loss = 0.5 * (self.gan_loss(d1_real, True, True) + self.gan_loss(d1_fake, False, True))
            d2_loss = 0.5 * (self.gan_loss(d2_real, True, True) + self.gan_loss(d2_fake, False, True))
        
        g1_gradients = tape.gradient(g1_total_loss, self.G1.trainable_variables)
        g2_gradients = tape.gradient(g2_total_loss, self.G2.trainable_variables)
        
        self.g1_optimizer.apply_gradients(zip(g1_gradients, self.G1.trainable_variables))
        self.g2_optimizer.apply_gradients(zip(g2_gradients, self.G2.trainable_variables))
        
        d1_gradients = tape.gradient(d1_loss, self.D1.trainable_variables)
        d2_gradients = tape.gradient(d2_loss, self.D2.trainable_variables)
        
        self.d1_optimizer.apply_gradients(zip(d1_gradients, self.D1.trainable_variables))
        self.d2_optimizer.apply_gradients(zip(d2_gradients, self.D2.trainable_variables))
        
        del tape
        
        return {
            'g1_total_loss': g1_total_loss,
            'g2_total_loss': g2_total_loss,
            'd1_loss': d1_loss,
            'd2_loss': d2_loss
        }
    
    def train(self, epochs):
        """Training loop for the CGAN"""
        dataset = ISTDDataset(self.args.data_dir[0], batch_size=self.args.batch_size)
        self.train_dataset = dataset.get_train_dataset()
        
        for epoch in range(epochs):
            self.logger.info(f'Starting epoch {epoch+1}/{epochs}')
            
            g_losses = []
            d_losses = []
            
            for batch in self.train_dataset:
                losses_dict = self.train_step(batch)
                g_losses.append(losses_dict['g1_total_loss'] + losses_dict['g2_total_loss'])
                d_losses.append(losses_dict['d1_loss'] + losses_dict['d2_loss'])
            
            g_loss = tf.reduce_mean(g_losses)
            d_loss = tf.reduce_mean(d_losses)
            
            self.logger.info(f'Epoch {epoch+1}/{epochs}')
            self.logger.info(f'Generator Loss: {g_loss:.4f}')
            self.logger.info(f'Discriminator Loss: {d_loss:.4f}')
            
            if (epoch + 1) % 10 == 0:
                checkpoint_dir = os.path.join(self.args.weights, f'checkpoint_epoch_{epoch+1}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                self.G1.save_weights(os.path.join(checkpoint_dir, 'G1.h5'))
                self.G2.save_weights(os.path.join(checkpoint_dir, 'G2.h5'))
                self.D1.save_weights(os.path.join(checkpoint_dir, 'D1.h5'))
                self.D2.save_weights(os.path.join(checkpoint_dir, 'D2.h5'))
                
                self.logger.info(f'Saved checkpoint at epoch {epoch+1}')
                
    def train_with_progress_bar(self, epochs):
        """Train the model for the specified number of epochs with progress bar"""
        dataset = ISTDDataset(self.args.data_dir[0], batch_size=self.args.batch_size)
        self.train_dataset = dataset.get_train_dataset()
        
        n_batches = len(dataset)
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            metrics = {
                'g1_total_loss': 0.0,
                'g2_total_loss': 0.0,
                'd1_loss': 0.0,
                'd2_loss': 0.0
            }
            
            with tqdm(total=n_batches, desc=f'Epoch {epoch}/{epochs}') as pbar:
                for batch_idx, batch in enumerate(self.train_dataset):
                    losses = self.train_step(batch)
                    
                    for k, v in losses.items():
                        metrics[k] += v.numpy()
                    
                    pbar.set_postfix({
                        'G1': f"{losses['g1_total_loss']:.4f}",
                        'G2': f"{losses['g2_total_loss']:.4f}",
                        'D1': f"{losses['d1_loss']:.4f}",
                        'D2': f"{losses['d2_loss']:.4f}"
                    })
                    pbar.update(1)
            
            for k in metrics:
                metrics[k] /= n_batches
            
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch}/{epochs} - {epoch_time:.2f}s - "
                f"G1: {metrics['g1_total_loss']:.4f}, "
                f"G2: {metrics['g2_total_loss']:.4f}, "
                f"D1: {metrics['d1_loss']:.4f}, "
                f"D2: {metrics['d2_loss']:.4f}"
            )
            
            if epoch % 5 == 0:
                self._save_weights(epoch)
                
        self.logger.info("Training finished")
    
    def _save_weights(self, epoch):
        """Save model weights"""
        checkpoint_dir = os.path.join(self.args.weights, f'epoch_{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.G1.save_weights(os.path.join(checkpoint_dir, 'G1.h5'))
        self.G2.save_weights(os.path.join(checkpoint_dir, 'G2.h5'))
        self.D1.save_weights(os.path.join(checkpoint_dir, 'D1.h5'))
        self.D2.save_weights(os.path.join(checkpoint_dir, 'D2.h5'))
        
        self.logger.info(f'Saved weights at epoch {epoch}')
    
    def infer(self, image):
        """
        Perform shadow removal inference on a single image
        
        Args:
            image (tf.Tensor): Input image tensor of shape [H, W, 3]
            
        Returns:
            tuple: (shadow_mask, shadow_free_image)
        """
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image)
            
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
            
        shadow_mask = self.G1(image, training=False)
        
        shadow_free = self.G2(tf.concat([image, shadow_mask], axis=-1), training=False)
        
        if len(image.shape) == 4 and image.shape[0] == 1:
            shadow_mask = tf.squeeze(shadow_mask, 0)
            shadow_free = tf.squeeze(shadow_free, 0)
            
        return shadow_mask, shadow_free
    
    def save(self, path):
        """Save model checkpoints"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.G1.save_weights(path + '_G1.h5')
        self.G2.save_weights(path + '_G2.h5')
        self.D1.save_weights(path + '_D1.h5')
        self.D2.save_weights(path + '_D2.h5')
    
    def load(self, path):
        """Load model checkpoints"""
        self.G1.load_weights(path + '_G1.h5')
        self.G2.load_weights(path + '_G2.h5')
        self.D1.load_weights(path + '_D1.h5')
        self.D2.load_weights(path + '_D2.h5')
