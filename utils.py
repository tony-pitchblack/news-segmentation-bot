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
    Sets up a logger with optional colorized `levelname` output.

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
            # Define log colors for `levelname` and `funcName`
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)s%(reset)s:"
                "%(blue)s%(funcName)s%(reset)s:"
                "%(lineno)d:%(message)s",
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
        formatter = logging.Formatter("%(levelname)s:%(funcName)s:%(lineno)d:%(message)s")
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger

def setup_file_logger(file_path, logger_name=None, log_level=logging.INFO):
    """
    Sets up a file logger to write raw text logs to a specified file.
    
    Parameters:
        file_path (str): Path to the log file.
        logger_name (str): Name of the logger.
        log_level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setLevel(log_level)

    # Raw text format for file logs
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger

from types import SimpleNamespace

REPO_DIRS = SimpleNamespace(
    logs='./logs',
    segmentation_logs='./logs/segmentation'
)

import os

def check_and_make_directories():
    for _dir in REPO_DIRS.__dict__.values():
        os.makedirs(_dir, exist_ok=True)