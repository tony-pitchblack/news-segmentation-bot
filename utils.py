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

from models.lightning_model import TextSegmenter
from models.EncoderDataset import Predictor
import wandb
import os

def load_model_from_wandb(run_id='k4j7vuo7'):
    api = wandb.Api()
    artifact = api.artifact(f'overfit1010/lenta_BiLSTM_F1/model-{run_id}:v0', type='model')
    art_dir = artifact.download()
    ckpt_path = os.path.join(art_dir, 'model.ckpt')

    text_seg_model = TextSegmenter.load_from_checkpoint(ckpt_path).to('cpu')
    predictor_model = Predictor(text_seg_model, sentence_encoder="cointegrated/rubert-tiny2")

    return predictor_model