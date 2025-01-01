from utils import setup_logger, format_time
from pathlib import Path
import logging
import json

logger = setup_logger(
    Path(__file__).stem, 
    log_level=logging.INFO,
    # log_level=logging.DEBUG
)

import asyncio
import os
import numpy as np


# If there are sentences, prepare them for segmentation inference
async def detect_segment_boundaries(sentence_generator, predictor_model, batch_size=1):
    async def generator():
        sentences = []
        async for sentence in sentence_generator:
            sentences.append(sentence)
            if len(sentences) >= batch_size:
                # Predict boundaries using the text segmentation model
                doc = [sentence['text'] for sentence in sentences]
                predictions = predictor_model.predict([doc], pretokenized_sents=[doc])
                boundary_mask = predictions[0]['boundaries'][0]
                boundary_mask = np.array(boundary_mask)
                yield (sentences, boundary_mask)

                sentences = []
    return generator()

async def handle_segments(boundary_generator):
    async for sentences, boundary_mask in boundary_generator:
        # Print boundary prediction for each sentence
        for idx, (sentence, boundary_flag) in enumerate(zip(sentences, boundary_mask)):
            text = sentence['text']
            start = sentence['start']
            end = sentence['end']

            # Handle missing timestamps with fallback logic
            if start is None:
                start = prev_start + 1  # Estimate start
            if end is None:
                end = start + 1  # Estimate end

            prev_start = start

            # Print the sentence with its boundary status
            print(
                f"[{format_time(start)} - {format_time(end)}] ",
                "BOUNDARY: " if boundary_flag else "INNER: ",
                text,
                sep=''
            )


async def transcribe_audio_stream(stream_url, step_s, model, language, max_duration, verbosity, print_openai, whisper_cpp_root_path):
    logger.info("Starting audio transcribation...")
    command = f"""
    cd {whisper_cpp_root_path}
    ./examples/livestream.sh "{stream_url}" {str(step_s)} {model} {language} {str(max_duration)} {str(verbosity)} {str(print_openai)}
    """

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def generator():
        # Read from stdout and stderr asynchronously
        async def read_stream(stream, log_fn=None):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line = line.decode().strip()
                log_fn('\n' + line)
                yield line

        # Yield from stdout while logging stderr
        async for line in read_stream(process.stdout, log_fn=logger.debug):
            if line != "":
                transcribed_segment = json.loads(line)
                yield transcribed_segment

        async for line in read_stream(process.stderr, log_fn=logger.error):
            pass

        # Wait for the process to finish
        await process.wait()

    # Return the constructed async generator
    return generator()

async def generate_sentences(transcript_generator):
    async def sentence_generator():
        sentence_buffer = []
        start_time = None

        async for transcript in transcript_generator:
            # Append new text to the buffer
            sentence_buffer.append(transcript["text"])
            
            # Set the start time of the first segment
            if start_time is None:
                start_time = transcript["start"]

            # Check if the buffer ends with a complete sentence
            current_text = " ".join(sentence_buffer).strip()
            if current_text.endswith((".", "!", "?")):
                # Yield the complete sentence
                yield {
                    "start": start_time,
                    "end": transcript["end"],
                    "text": current_text
                }
                # Reset the buffer and start time
                sentence_buffer = []
                start_time = None

        # Handle any remaining text in the buffer after the generator ends
        if sentence_buffer:
            yield {
                "start": start_time,
                "end": transcript["end"],
                "text": " ".join(sentence_buffer).strip()
            }

    return sentence_generator()

def load_model_from_wandb(run_id='k4j7vuo7'):
    from nse_topic_segmentation.models.lightning_model import TextSegmenter
    from nse_topic_segmentation.models.EncoderDataset import Predictor
    import wandb

    logger.info('Loading model...')
    api = wandb.Api()
    artifact = api.artifact(f'overfit1010/lenta_BiLSTM_F1/model-{run_id}:v0', type='model')
    art_dir = artifact.download()
    ckpt_path = os.path.join(art_dir, 'model.ckpt')

    text_seg_model = TextSegmenter.load_from_checkpoint(ckpt_path).to('cpu')
    predictor_model = Predictor(text_seg_model, sentence_encoder="cointegrated/rubert-tiny2")

    return predictor_model

from dotenv import load_dotenv

async def main():
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run the livestream with the specified parameters.")
    
    # Define required positional argument for the stream URL
    parser.add_argument("--stream_url", type=str, default=None, help="The URL of the stream.")
    
    # Define optional arguments with defaults, now in long-form (optional with flags)
    parser.add_argument("--step_s", type=int, default=15, help="Step in seconds for the stream.")
    parser.add_argument("--model", type=str, default="small", help="Model to use.")
    parser.add_argument("--language", type=str, default="ru", help="Language of the stream.")
    parser.add_argument("--max_duration", type=int, default=60, help="Maximum duration for the stream.")
    parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level.")
    parser.add_argument("--print_openai", type=int, default=1, help="Whether to print OpenAI output.")
    parser.add_argument("--whisper_cpp_root_path", type=str, default='../whisper.cpp', help="whisper.cpp root path.")
    
    # Parse the arguments
    args = parser.parse_args()

    if args.stream_url is None:
        logger.info("Stream URL is not provided, loading STREAM_URL from `configs/news_url.env`")
        load_dotenv("configs/stream_url.env")
        STREAM_URL = os.getenv("STREAM_URL")

    # suppress_verbose_logging()

    load_dotenv("configs/keys.env") # for WANDB_API_KEY
    HF_HOME = os.getenv("HF_HOME")
    if HF_HOME is None:
        HF_HOME = "~/.cache/huggingface"

    HF_HOME = os.path.expanduser(HF_HOME)
    if not (os.access(HF_HOME, os.R_OK) and os.access(HF_HOME, os.W_OK)):
        raise PermissionError(
            f"{HF_HOME} doesn't have read and write permissions for the current user.\n"
            "Either change permissions or set another HF_HOME directory in configs/whisper-cpp-client.env"
        )

    predictor_model = load_model_from_wandb()

    kwargs = dict(
        stream_url=STREAM_URL,
        step_s=15,
        model="small",
        language="ru",
        max_duration=30,
        verbosity=0,
        print_openai=1,
        whisper_cpp_root_path="../whisper.cpp"
    )

    # await transcribe_audio_stream(
    #     **kwargs
    # )

    transcript_generator = await transcribe_audio_stream(**kwargs)
    # async for line in transcript_generator:
    #     print(line)

    sentence_generator = await generate_sentences(transcript_generator)
    # async for line in sentence_generator:
    #     print(line)

    # await detect_segment_boundaries(sentence_generator, predictor_model=predictor_model)
    boundary_generator = await detect_segment_boundaries(sentence_generator, predictor_model=predictor_model)
    # async for line in boundary_generator:
    #     print(line)

    await handle_segments(boundary_generator)

if __name__ == "__main__":
    asyncio.run(main())