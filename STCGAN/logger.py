#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced logging utilities with TensorFlow integration and tqdm support
"""

import logging
import sys
import io

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm, trange


class TeeIo:
    """
    A file-like object that writes to both a file and a stream simultaneously
    Mimics the behavior of the Unix 'tee' command
    """

    def __init__(self, file, stream=sys.stderr):
        """
        Initialize TeeIo
        
        Args:
            file (str): Path to log file
            stream (file-like, optional): Output stream. Defaults to sys.stderr.
        """
        self.file = open(file, 'w', buffering=1)
        self.stream = stream

    def close(self):
        """Close the file and stream"""
        self.file.close()

    def write(self, data, to_stream=True):
        """
        Write data to file and optionally to stream
        
        Args:
            data (str): Data to write
            to_stream (bool, optional): Whether to write to stream. Defaults to True.
        """
        self.file.write(data)
        if to_stream:
            self.stream.write(data)

    def flush(self):
        """Flush both file and stream buffers"""
        self.file.flush()
        self.stream.flush()


class TqdmStreamHandler(logging.StreamHandler):
    """
    A logging StreamHandler that uses tqdm.write() for output
    Ensures logging messages don't interfere with tqdm progress bars
    """

    def emit(self, record):
        """
        Emit a log record using tqdm.write()
        
        Args:
            record (logging.LogRecord): Log record to emit
        """
        try:
            msg = self.format(record)
            return tqdm.write(msg, file=self.stream)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
        return super().emit(record)


class Logger:
    """
    Advanced logging utility with file and console logging, 
    TensorFlow summary writing, and tqdm integration
    """

    def __init__(self, log_file, log_dir=None, level=logging.INFO):
        """
        Initialize Logger
        
        Args:
            log_file (str): Path to log file
            log_dir (str, optional): TensorFlow summary log directory
            level (int, optional): Logging level. Defaults to logging.INFO.
        """
        self.stream = TeeIo(log_file, stream=sys.stderr)
        self.logger = self._create_logger(log_file, level=level)
        
        # TensorFlow summary writer
        self.summary_writer = tf.summary.create_file_writer(log_dir) if log_dir else None

    def _create_logger(self, 
                       log_file, 
                       level=logging.DEBUG, 
                       file_level=None, 
                       console_level=None):
        """
        Create a logger with custom handlers
        
        Args:
            log_file (str): Path to log file
            level (int, optional): Base logging level
            file_level (int, optional): File logging level
            console_level (int, optional): Console logging level
        
        Returns:
            logging.Logger: Configured logger
        """
        file_level = level if file_level is None else file_level
        console_level = level if console_level is None else console_level

        logger = logging.getLogger(log_file)
        logger.setLevel(level)
        
        # Create file handler
        fh = TqdmStreamHandler(self.stream)
        fh.setLevel(file_level)
        
        # Create formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(filename)-15s %(levelname)-6s %(message)s',
            datefmt="%H:%M:%S",
            style='%')
        fh.setFormatter(file_formatter)
        
        # Add handler to logger
        logger.addHandler(fh)
        return logger

    def __del__(self):
        """Close stream on object deletion"""
        self.stream.close()
        if self.summary_writer:
            self.summary_writer.close()

    def tqdm(self, *args, **kwargs):
        """
        Create a tqdm progress bar with custom stream
        
        Returns:
            tqdm: Progress bar
        """
        kwargs["file"] = self.stream
        return tqdm(*args, **kwargs)

    def trange(self, *args, **kwargs):
        """
        Create a tqdm range with custom stream
        
        Returns:
            tqdm: Progress range
        """
        kwargs["file"] = self.stream
        return trange(*args, **kwargs)

    def scalar_summary(self, tag, value, step):
        """
        Log scalar value to TensorFlow summary
        
        Args:
            tag (str): Summary tag
            value (float): Scalar value
            step (int): Training step
        """
        if self.summary_writer:
            with self.summary_writer.as_default():
                tf.summary.scalar(tag, value, step=step)

    def image_summary(self, tag, image, step):
        """
        Log image to TensorFlow summary
        
        Args:
            tag (str): Summary tag
            image (np.ndarray): Image to log
            step (int): Training step
        """
        if self.summary_writer:
            # Ensure image is in the right format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            with self.summary_writer.as_default():
                tf.summary.image(tag, 
                                 tf.expand_dims(image, 0),  # Add batch dimension
                                 step=step)

    def image_list_summary(self, tag, images, step):
        """
        Log multiple images to TensorFlow summary
        
        Args:
            tag (str): Base summary tag
            images (list): List of images to log
            step (int): Training step
        """
        if not images or not self.summary_writer:
            return
        
        # Ensure images are in the right format
        processed_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            processed_images.append(img)
        
        with self.summary_writer.as_default():
            for i, img in enumerate(processed_images):
                tf.summary.image(
                    f"{tag}/{i}", 
                    tf.expand_dims(img, 0),  # Add batch dimension
                    step=step
                )
