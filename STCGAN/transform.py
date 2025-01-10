import numbers

import cv2 as cv
import numpy as np
import tensorflow as tf


def transforms(resize=None,
               scale=None,
               angle=None,
               flip_prob=None,
               crop_size=None):
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
    """
    Compose multiple image transformations
    """
    def __init__(self, transforms: list):
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
            sample (np.ndarray): Input images or arrays
        
        Returns:
            tuple or np.ndarray: Transformed images
        """
        for transform in self.transforms:
            sample = transform(*sample)
        return sample


class Normalize:
    """
    Normalize images by subtracting mean and dividing by standard deviation
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        """
        Initialize normalization parameters
        
        Args:
            mean (tuple, optional): Mean values for each channel
            std (tuple, optional): Standard deviation values for each channel
        """
        self.mean = np.array(mean).reshape(-1)
        self.std = np.array(std).reshape(-1)

    def __call__(self, *datas, inverse=False):
        """
        Normalize or denormalize images
        
        Args:
            datas (np.ndarray): Input images
            inverse (bool, optional): Whether to denormalize
        
        Returns:
            np.ndarray or list: Normalized/denormalized images
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
    """
    Randomly scale images
    """
    def __init__(self, scale):
        """
        Initialize random scaling
        
        Args:
            scale (float): Maximum scaling factor
        """
        assert 0 <= scale and scale <= 0.5
        self.scale = scale

    def __call__(self, *datas):
        """
        Apply random scaling to images
        
        Args:
            datas (np.ndarray): Input images
        
        Returns:
            np.ndarray or list: Scaled images
        """
        outputs = []
        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)
        interp = cv.INTER_LINEAR if scale > 1 else cv.INTER_AREA
        
        for x in datas:
            rows, cols = x.shape[:2]
            M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 0, scale)
            outputs.append(cv.warpAffine(x, M, (cols, rows),
                                         flags=interp,
                                         borderMode=cv.BORDER_CONSTANT))
        
        return outputs if len(datas) > 1 else outputs[0]


class RandomRotate:
    """
    Randomly rotate images
    """
    def __init__(self, angle):
        """
        Initialize random rotation
        
        Args:
            angle (float): Maximum rotation angle
        """
        self.angle = angle

    def __call__(self, *datas):
        """
        Apply random rotation to images
        
        Args:
            datas (np.ndarray): Input images
        
        Returns:
            np.ndarray or list: Rotated images
        """
        outputs = []
        angle = np.random.uniform(low=-self.angle, high=self.angle)
        
        for x in datas:
            rows, cols = x.shape[:2]
            M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)
            outputs.append(cv.warpAffine(x, M, (cols, rows),
                                         borderMode=cv.BORDER_CONSTANT))
        
        return outputs if len(datas) > 1 else outputs[0]


class RandomHorizontalFlip:
    """
    Randomly flip images horizontally
    """
    def __init__(self, flip_prob):
        """
        Initialize horizontal flip
        
        Args:
            flip_prob (float): Probability of flipping
        """
        self.flip_prob = flip_prob

    def __call__(self, *datas):
        """
        Apply random horizontal flip to images
        
        Args:
            datas (np.ndarray): Input images
        
        Returns:
            np.ndarray or list: Flipped or original images
        """
        if np.random.rand() > self.flip_prob:
            return datas
        else:
            outputs = [np.fliplr(x).copy() for x in datas]
            return outputs if len(datas) > 1 else outputs[0]


class RandomCrop:
    """
    Randomly crop images
    """
    def __init__(self, size):
        """
        Initialize random cropping
        
        Args:
            size (int or tuple): Target crop size
        """
        if isinstance(size, numbers.Number):
            to_size = (int(size), int(size))
        else:
            to_size = size
        self.rows, self.cols = to_size

    def __call__(self, *datas):
        """
        Apply random cropping to images
        
        Args:
            datas (np.ndarray): Input images
        
        Returns:
            np.ndarray or list: Cropped images
        """
        rows, cols = datas[0].shape[:2]  # datas should have the same size
        padding = self.rows > rows or self.cols > cols
        
        if padding:
            # padding is needed if the target size is larger than the image
            pad_height = max((self.rows - rows), 0)
            pad_width = max((self.cols - cols), 0)
            rows += 2*pad_height
            cols += 2*pad_width

        row_offset = np.random.randint(low=0, high=(rows-self.rows))
        col_offset = np.random.randint(low=0, high=(cols-self.cols))

        outputs = []
        for x in datas:
            if padding:
                x = cv.copyMakeBorder(x,
                                      pad_height, pad_height,
                                      pad_width, pad_width,
                                      cv.BORDER_CONSTANT, value=0)
            outputs.append(x[row_offset:row_offset+self.rows,
                             col_offset:col_offset+self.cols, ...].copy())
        
        return outputs if len(datas) > 1 else outputs[0]


class Resize:
    """
    Resize images to specified dimensions
    """
    def __init__(self, size):
        """
        Initialize resizing
        
        Args:
            size (int or tuple): Target resize dimensions
        """
        if isinstance(size, numbers.Number):
            to_size = (int(size), int(size))
        else:
            to_size = size
        self.rows, self.cols = to_size

    def __call__(self, *datas):
        """
        Apply resizing to images
        
        Args:
            datas (np.ndarray): Input images
        
        Returns:
            np.ndarray or list: Resized images
        """
        outputs = []
        for x in datas:
            rows, cols = x.shape[:2]
            
            if self.rows < rows and self.cols < cols:
                interp = cv.INTER_AREA
            else:
                interp = cv.INTER_LINEAR
            
            outputs.append(
                cv.resize(x, (self.cols, self.rows), interpolation=interp))
        
        return outputs if len(datas) > 1 else outputs[0]
