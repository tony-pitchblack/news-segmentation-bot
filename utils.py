from urllib.parse import urlencode, quote

def get_whisper_server_url(host, model, language="ru", response_format="verbose_json", port=8000, escaped=False):
    # Define query parameters
    options = {
        "language": language,
        "model": model,
        "response_format": response_format
    }

    # Base WebSocket URL
    base_ws_url = f"ws://{host}:{port}/v1/audio/transcriptions"

    # Construct the full URL with encoded query parameters
    whisper_server_url = f"{base_ws_url}?{urlencode(options, quote_via=quote)}"

    # Escape ampersands if requested
    if escaped:
        whisper_server_url = whisper_server_url.replace('&', r'\&')

    return whisper_server_url


import time
from datetime import timedelta

def measure_execution_time(func):
    """Decorator to measure the execution time of a function."""
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the timer
        result = await func(*args, **kwargs)  # Call the function
        end_time = time.perf_counter()  # End the timer
        execution_time = end_time - start_time  # Calculate the execution time
        print(f"Execution time for {func.__name__}: {timedelta(seconds=execution_time)}")
        return result

    return wrapper

def format_time(seconds):
    """
    Format time in "hh:mm:ss.xx"
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs % 1) * 100)
    secs = int(secs)  # Remove fractional part for formatting
    # return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:02}"
    return f"{hours:02}:{minutes:02}:{secs:02}"


import logging
import colorlog

def setup_logger(logger_name=None, log_level=logging.INFO, color=True):
    """
    Sets up a logger with optional colorized `levelname` output and default time formatting.

    Parameters:
        logger_name (str): Name of the logger.
        log_level (int): Logging level.
        color (bool): If True, enable colorized `levelname` in the logs.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if color:
        # Formatter with color and default time formatting
        formatter = colorlog.ColoredFormatter(
            "[%(asctime)s] "
            "%(log_color)s%(levelname)s%(reset)s:"
            "%(blue)s%(funcName)s%(reset)s:"
            "%(lineno)d: %(message)s",
            datefmt="%H:%M:%S",  # Default time format
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        )
    else:
        # Standard formatter without colors
        formatter = logging.Formatter(
            "%(levelname)s:%(asctime)s:%(funcName)s:%(lineno)d:%(message)s",
            datefmt="%H:%M:%S"
        )
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger

class PrefixFilter(logging.Filter):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def filter(self, record):
        record.msg = f"{self.prefix} {record.msg}"
        return True

def setup_file_logger(file_path, logger_name=None, log_level=logging.INFO, log_prefix='only_ts'):
    """
    Sets up a file logger to write raw text logs to a specified file with optional prefix.
    
    Parameters:
        file_path (str): Path to the log file.
        logger_name (str): Name of the logger.
        log_level (int): Logging level.
        log_prefix (bool): If True, add a log prefix to the message.

    Returns:
        logging.Logger: Configured logger instance.
    """
    assert isinstance(log_prefix, bool) or log_prefix == 'only_ts' 

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Create file handler
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setLevel(log_level)

    # If log_prefix is True, add prefix similar to the vanilla setup_logger()
    if log_prefix:
        formatter_string = "[%(asctime)s]"

        if log_prefix != 'only_ts':
            formatter_string += "%(levelname)s:%(funcName)s:%(lineno)d:"
        
        formatter_string +=  " %(message)s"
        
        file_formatter = logging.Formatter(
            formatter_string,
            datefmt="%H:%M:%S"
        )
    else:
        # Standard formatter for file logs
        file_formatter = logging.Formatter('%(message)s')

    file_handler.setFormatter(file_formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
    
from types import SimpleNamespace

REPO_DIRS = SimpleNamespace(
    logs_dir='./logs',
)

import os

def check_and_make_directories():
    for _dir in REPO_DIRS.__dict__.values():
        os.makedirs(_dir, exist_ok=True)