"""
Advanced logging utilities with tqdm integration
"""

import logging
import sys
from typing import Optional, Union

from tqdm.auto import tqdm


class TeeIo:
    """
    A file-like object that writes to both a file and a stream simultaneously
    
    Attributes:
        file (file): Output file
        stream (file): Output stream (default: sys.stderr)
    """

    def __init__(self, 
                 file_path: str, 
                 stream: Optional[Union[None, sys._FileType]] = sys.stderr):
        """
        Initialize TeeIo
        
        Args:
            file_path (str): Path to log file
            stream (file, optional): Stream to write to. Defaults to sys.stderr.
        """
        self.file = open(file_path, 'w', buffering=1)
        self.stream = stream

    def close(self):
        """Close the file"""
        self.file.close()

    def write(self, data: str, to_stream: bool = True):
        """
        Write data to file and optionally to stream
        
        Args:
            data (str): Data to write
            to_stream (bool, optional): Whether to write to stream. Defaults to True.
        """
        self.file.write(data)
        if to_stream and self.stream:
            self.stream.write(data)

    def flush(self):
        """Flush both file and stream buffers"""
        self.file.flush()
        if self.stream:
            self.stream.flush()


class TqdmStreamHandler(logging.StreamHandler):
    """
    A logging StreamHandler that uses tqdm.write() for output
    
    Ensures logging messages are displayed correctly with tqdm progress bars
    """

    def __init__(self, stream=sys.stderr):
        """
        Initialize TqdmStreamHandler
        
        Args:
            stream (file, optional): Output stream. Defaults to sys.stderr.
        """
        super().__init__(stream)

    def emit(self, record):
        """
        Emit a log record using tqdm.write()
        
        Args:
            record (logging.LogRecord): Log record to emit
        """
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class Logger:
    """
    Advanced logging utility with file and console output
    
    Supports tqdm integration and flexible logging configurations
    """

    def __init__(self, 
                 log_file: str, 
                 level: int = logging.INFO, 
                 *args, **kwargs):
        """
        Initialize Logger
        
        Args:
            log_file (str): Path to log file
            level (int, optional): Logging level. Defaults to logging.INFO.
        """
        self.stream = TeeIo(log_file, stream=sys.stderr)
        self.logger = self._create_logger(
            log_file, 
            level=level, 
            *args, 
            **kwargs
        )

    def _create_logger(self,
                       log_file: str,
                       level: int = logging.DEBUG,
                       file_level: Optional[int] = None,
                       console_level: Optional[int] = None) -> logging.Logger:
        """
        Create a configured logger
        
        Args:
            log_file (str): Path to log file
            level (int, optional): Base logging level. Defaults to logging.DEBUG.
            file_level (int, optional): File logging level. Defaults to base level.
            console_level (int, optional): Console logging level. Defaults to base level.
        
        Returns:
            logging.Logger: Configured logger
        """
        file_level = level if file_level is None else file_level
        console_level = level if console_level is None else console_level

        logger = logging.getLogger(log_file)
        logger.setLevel(level)

        fh = TqdmStreamHandler(self.stream)
        fh.setLevel(file_level)

        file_formatter = logging.Formatter(
            '%(asctime)s - %(filename)-15s %(levelname)-6s %(message)s',
            datefmt="%H:%M:%S",
            style='%'
        )
        fh.setFormatter(file_formatter)

        logger.addHandler(fh)

        return logger

    def __getattr__(self, name):
        """
        Delegate method calls to the underlying logger
        
        Args:
            name (str): Method name
        
        Returns:
            Callable: Logger method
        """
        return getattr(self.logger, name)
