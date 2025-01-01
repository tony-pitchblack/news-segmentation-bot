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

import logging

def setup_logger(name=None, log_level=logging.INFO):
    logging.basicConfig(format="%(levelname)s:%(funcName)s:%(lineno)d:%(message)s")
    # logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s")
    logger = logging.getLogger(name=name)
    logger.setLevel(log_level)
    return logger

# Function to format time in "hh:mm:ss.xx"
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs % 1) * 100)
    secs = int(secs)  # Remove fractional part for formatting
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:02}"