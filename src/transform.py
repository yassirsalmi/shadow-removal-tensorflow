import numbers
from typing import Union, Tuple, List, Optional

import tensorflow as tf
import numpy as np


def transforms(
    resize: Optional[Union[int, Tuple[int, int]]] = None,
    scale: Optional[float] = None,
    angle: Optional[float] = None,
    flip_prob: Optional[float] = None,
    crop_size: Optional[Union[int, Tuple[int, int]]] = None
) -> 'Compose':
    """
    Create a composition of image transformations
    
    Args:
        resize (int or tuple, optional): Resize dimensions
        scale (float, optional): Random scaling factor
        angle (float, optional): Random rotation angle
        flip_prob (float, optional): Horizontal flip probability
        crop_size (int or tuple, optional): Random crop size
    
    Returns:
        Compose: Composed image transformations
    """
    transform_list = []
    if resize is not None:
        transform_list.append(Resize(resize))
    if scale is not None:
        transform_list.append(RandomScale(scale))
    if angle is not None:
        transform_list.append(RandomRotate(angle))
    if flip_prob is not None:
        transform_list.append(RandomHorizontalFlip(flip_prob))
    if crop_size is not None:
        transform_list.append(RandomCrop(crop_size))

    return Compose(transform_list)


class Compose:
    """Compose multiple image transformations"""
    def __init__(self, transforms: List):
        """
        Initialize composition of transformations
        
        Args:
            transforms (list): List of transformation functions
        """
        self.transforms = transforms

    def __call__(self, *sample):
        """
        Apply transformations sequentially
        
        Args:
            sample: Input image(s)
        
        Returns:
            Transformed image(s)
        """
        for transform in self.transforms:
            sample = transform(*sample)
        return sample


class Normalize:
    """Normalize images using mean and standard deviation"""
    def __init__(self, 
                 mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 std: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        """
        Initialize normalization parameters
        
        Args:
            mean (tuple): Mean values for each channel
            std (tuple): Standard deviation values for each channel
        """
        self.mean = np.array(mean).reshape(-1)
        self.std = np.array(std).reshape(-1)

    def __call__(self, *datas, inverse: bool = False):
        """
        Normalize or denormalize images
        
        Args:
            datas: Input images
            inverse (bool, optional): Whether to denormalize
        
        Returns:
            Normalized or denormalized images
        """
        outputs = []
        for x in datas:
            assert x.shape[-1] == len(self.mean)
            assert x.shape[-1] == len(self.std)
            
            if not inverse:
                y = (x - self.mean) / self.std
            else:
                y = (x * self.std) + self.mean
            
            outputs.append(y)
        
        return outputs if len(datas) > 1 else outputs[0]


class RandomScale:
    """Randomly scale images with PyTorch-like behavior"""
    def __init__(self, scale: float):
        """
        Initialize random scaling
        
        Args:
            scale (float): Maximum scale factor (0 <= scale <= 0.5)
        """
        assert 0 <= scale <= 0.5
        self.scale = scale

    def __call__(self, *datas):
        """
        Apply random scaling to images with PyTorch-like interpolation
        
        Args:
            datas: Input images
        
        Returns:
            Scaled images
        """
        outputs = []
        scale = tf.random.uniform([], minval=1.0 - self.scale, maxval=1.0 + self.scale)
        for x in datas:
            # Get original dimensions
            height, width = tf.shape(x)[0], tf.shape(x)[1]
            
            # Calculate new dimensions
            new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
            
            # Resize with bilinear interpolation (matches PyTorch's behavior)
            resized = tf.image.resize(x, [new_height, new_width], 
                                    method=tf.image.ResizeMethod.BILINEAR,
                                    antialias=True,
                                    preserve_aspect_ratio=True)
            
            # Center crop/pad to original size (with reflection padding)
            if scale > 1:
                # Center crop
                start_h = (new_height - height) // 2
                start_w = (new_width - width) // 2
                resized = resized[start_h:start_h+height, start_w:start_w+width]
            else:
                # Reflection pad (matches PyTorch's behavior)
                pad_h = (height - new_height) // 2
                pad_w = (width - new_width) // 2
                resized = tf.pad(resized, 
                               [[pad_h, height-new_height-pad_h], 
                                [pad_w, width-new_width-pad_w], 
                                [0, 0]], 
                               mode='REFLECT')
            
            outputs.append(resized)
            
        return outputs if len(datas) > 1 else outputs[0]


class RandomRotate:
    """Randomly rotate images with PyTorch-like behavior"""
    def __init__(self, angle: float):
        """
        Initialize random rotation
        
        Args:
            angle (float): Maximum rotation angle
        """
        self.angle = angle

    def __call__(self, *datas):
        """
        Apply random rotation to images with PyTorch-like interpolation
        
        Args:
            datas: Input images
        
        Returns:
            Rotated images
        """
        outputs = []
        angle = tf.random.uniform([], minval=-self.angle, maxval=self.angle)
        radian = angle * np.pi / 180
        
        for x in datas:
            pad_size = int(max(x.shape[0], x.shape[1]) * 0.15)
            padded = tf.pad(x, 
                          [[pad_size, pad_size], 
                           [pad_size, pad_size], 
                           [0, 0]], 
                          mode='REFLECT')
            
            rotated = tf.image.rotate(padded, 
                                    radian, 
                                    interpolation='BILINEAR',
                                    fill_mode='REFLECT')
            
            height, width = x.shape[0], x.shape[1]
            start_h = (rotated.shape[0] - height) // 2
            start_w = (rotated.shape[1] - width) // 2
            cropped = rotated[start_h:start_h+height, start_w:start_w+width]
            
            outputs.append(cropped)
            
        return outputs if len(datas) > 1 else outputs[0]


class RandomHorizontalFlip:
    """Randomly flip images horizontally (matches PyTorch)"""
    def __init__(self, flip_prob: float = 0.5):
        """
        Initialize horizontal flip
        
        Args:
            flip_prob (float): Probability of flipping (default: 0.5)
        """
        self.flip_prob = flip_prob

    def __call__(self, *datas):
        """
        Apply random horizontal flip to images
        
        Args:
            datas: Input images
        
        Returns:
            Flipped or original images
        """
        outputs = []
        if tf.random.uniform([]) < self.flip_prob:
            for x in datas:
                outputs.append(tf.image.flip_left_right(x))
        else:
            outputs = list(datas)
            
        return outputs if len(datas) > 1 else outputs[0]


class RandomCrop:
    """Randomly crop images with PyTorch-like padding"""
    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = 0, pad_mode: str = 'reflect'):
        """
        Initialize random cropping
        
        Args:
            size (int or tuple): Target crop size
            padding (int): Optional padding before crop (default: 0)
            pad_mode (str): Padding mode ('reflect', 'constant', 'symmetric')
        """
        if isinstance(size, numbers.Number):
            to_size = (int(size), int(size))
        else:
            to_size = size
        self.rows, self.cols = to_size
        self.padding = padding
        self.pad_mode = pad_mode

    def __call__(self, *datas):
        """
        Apply random cropping to images with optional padding
        
        Args:
            datas: Input images
        
        Returns:
            Cropped images
        """
        outputs = []
        
        if self.padding > 0:
            padded_datas = []
            for x in datas:
                padded = tf.pad(x,
                              [[self.padding, self.padding],
                               [self.padding, self.padding],
                               [0, 0]],
                              mode=self.pad_mode.upper())
                padded_datas.append(padded)
            datas = padded_datas
        
        height, width = tf.shape(datas[0])[0], tf.shape(datas[0])[1]
        if height < self.rows or width < self.cols:
            raise ValueError(f"Crop size {(self.rows, self.cols)} is larger than image size {(height, width)}")
            
        top = tf.random.uniform([], 0, height - self.rows + 1, dtype=tf.int32)
        left = tf.random.uniform([], 0, width - self.cols + 1, dtype=tf.int32)
        
        for x in datas:
            cropped = tf.image.crop_to_bounding_box(x, top, left, self.rows, self.cols)
            outputs.append(cropped)
            
        return outputs if len(datas) > 1 else outputs[0]


class Resize:
    """Resize images with PyTorch-like interpolation"""
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: str = 'bilinear'):
        """
        Initialize image resizing
        
        Args:
            size (int or tuple): Target resize dimensions
            interpolation (str): Interpolation method ('bilinear', 'nearest', 'bicubic')
        """
        if isinstance(size, numbers.Number):
            to_size = (int(size), int(size))
        else:
            to_size = size
        self.rows, self.cols = to_size
        self.interpolation = interpolation

    def __call__(self, *datas):
        """
        Apply resizing to images with specified interpolation
        
        Args:
            datas: Input images
        
        Returns:
            Resized images
        """
        outputs = []
        for x in datas:
            if self.interpolation == 'nearest':
                method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            elif self.interpolation == 'bicubic':
                method = tf.image.ResizeMethod.BICUBIC
            else:  # default to bilinear
                method = tf.image.ResizeMethod.BILINEAR
                
            resized = tf.image.resize(x, [self.rows, self.cols],
                                    method=method,
                                    antialias=True,
                                    preserve_aspect_ratio=True)
            outputs.append(resized)
            
        return outputs if len(datas) > 1 else outputs[0]


def random_flip(sample: dict, prob: float = 0.5) -> dict:
    """Random horizontal flip matching PyTorch RandomHorizontalFlip"""
    if tf.random.uniform([]) < prob:
        return {k: tf.image.flip_left_right(v) for k, v in sample.items()}
    return sample


def random_crop(sample: dict, size: int) -> dict:
    """Random crop matching PyTorch RandomCrop"""
    shape = tf.shape(next(iter(sample.values())))
    height, width = shape[0], shape[1]
    
    max_x = width - size
    max_y = height - size
    
    x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
    y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    
    return {k: v[y:y+size, x:x+size] for k, v in sample.items()}


def random_rotate(sample: dict, angle: float) -> dict:
    """Random rotation matching PyTorch RandomRotation"""
    angle = tf.random.uniform([], -angle, angle, dtype=tf.float32)
    angle = angle * np.pi / 180.0
    
    return {k: tf.image.rotate(v, angle, interpolation='BILINEAR') 
            for k, v in sample.items()}


def resize(sample: dict, size: Union[int, Tuple[int, int]]) -> dict:
    """Resize matching PyTorch Resize"""
    if isinstance(size, int):
        size = (size, size)
    
    return {k: tf.image.resize(v, size, method='bilinear') 
            for k, v in sample.items()}


def normalize(sample: dict, 
             mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
             std: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> dict:
    """Normalize matching PyTorch Normalize"""
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)
    
    return {k: (v - mean) / std if v.shape[-1] == 3 else v 
            for k, v in sample.items()}
