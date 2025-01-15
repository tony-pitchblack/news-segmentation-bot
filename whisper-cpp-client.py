from utils import format_time
from utils import setup_logger, setup_file_logger
from utils import REPO_DIRS, check_and_make_directories
from pathlib import Path
import logging
import json
from glob import glob
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

# Setup logging
logger = setup_logger(
    Path(__file__).stem, 
    # log_level=logging.INFO,
    log_level=logging.DEBUG
)

check_and_make_directories()

current_datetime = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
current_logs_path = Path(REPO_DIRS.logs_dir) / current_datetime
os.makedirs(current_logs_path)

segmentation_logger = setup_file_logger(
    file_path= current_logs_path / f'segmentation.log', 
    logger_name='segmentation',
    log_level=logging.INFO,
    log_prefix=False
)

profiling_logger = setup_file_logger(
    file_path= current_logs_path / f'profiling.log', 
    logger_name='profiling',
    log_level=logging.INFO,
    log_prefix='only_ts'
)

# Set up async generator wrapper for logging time
import asyncio
import numpy as np
from time import perf_counter

async def timed_generator(generator):
    # print(generator.__name__)
    async for item in generator:
        start_time = perf_counter()  # Start timing

        # Yield the item
        yield item

        end_time = perf_counter()  # End timing
        execution_time = end_time - start_time
        # print(f"Execution time for a cycle: {execution_time:.4f} seconds")
        profiling_logger.info(f"({generator.__name__}) {execution_time:.4f} seconds")

# Main functions
async def generate_segments(sentence_generator, predictor_model, buffer_size=10):
    async def segment_generator():
        logger.debug('Generating another segment...')
        sentences = []
        segment_buffer = []

        async for sentence in sentence_generator:
            sentences.append(sentence)

            # Process in batches
            if len(sentences) >= buffer_size:
                # Predict boundaries using the text segmentation model
                doc = [sentence['text'] for sentence in sentences]
                predictions = predictor_model.predict([doc], pretokenized_sents=[doc])
                boundary_mask = predictions[0]['boundaries'][0]
                boundary_mask = np.array(boundary_mask)

                # Accumulate sentences into segments
                for sentence, boundary_flag in zip(sentences, boundary_mask):
                    segment_buffer.append(sentence)

                    # Yield the current segment if a boundary is detected
                    if boundary_flag:
                        yield segment_buffer
                        segment_buffer = []  # Reset buffer for next segment

                # Reset sentence buffer after processing the batch
                logger.debug('A segment has been generated.')
                logger.debug('Generating another segment...')
                sentences = []

        # Yield any remaining sentences as the final segment
        if segment_buffer:
            yield segment_buffer
            logger.debug('A segment has been generated.')

    return segment_generator()

import spacy
from keywords import check_symbols, exclude_words_up, find_keywords, keywords_up

nlp = spacy.load("ru_core_news_sm") # Load Russian model

def detect_keywords(text):
    # Применение лемматизации
    sentence_lemmatized = ' '.join([token.lemma_.upper() for token in nlp(text)])

    # Проверка наличия ключевых слов
    matched_raw = find_keywords(keywords_up, text, exclude_words_up)
    matched_lemm = find_keywords(keywords_up, sentence_lemmatized, exclude_words_up)

    return matched_raw | matched_lemm

async def classify_segments(segment_generator):
    logger.debug('Ready to classify segments.')
    end = seconds_since_midnight() # Let very first dummy segment end at current time
    async for segment_sentences in segment_generator:
        segment_duration = segment_sentences[-1]['end'] - segment_sentences[0]['start']
        logger.debug(f'Classifying another segment with duration {segment_duration} seconds...')

        first_sentence = segment_sentences[0]
        start = end
        end = start + first_sentence['end'] - first_sentence['start']

        keywords = detect_keywords(first_sentence['text'])
        print_sentence(first_sentence['text'], start, end, is_boundary_pred=True, keywords=keywords)

        for sentence in segment_sentences[1:]:
            keywords = detect_keywords(sentence['text'])
            
            start = end
            end = start + sentence['end'] - sentence['start']
            print_sentence(sentence['text'], start, end, keywords=keywords)

def seconds_since_midnight():
    now = datetime.now()
    midnight = datetime.combine(now.date(), time(0, 0, 0))

    return int((now - midnight).total_seconds())

def print_sentence(
        text,
        start,
        end,
        idx=None,
        is_boundary_pred=False, is_boundary_target=False,
        use_system_time=True,
        keywords=[]
    ):

    boundary_indicators = 'P' if is_boundary_pred else '-'
    boundary_indicators += 'T' if is_boundary_target else '-'
    # boundary_indicators += 'K' if keywords is not None else '-'

    prefix = f"{idx:03d}" if idx is not None else ''
    prefix = f"[{format_time(start)} - {format_time(end)}]"

    message_sentence = f'{prefix} {boundary_indicators} {text}'
    print(message_sentence)
    segmentation_logger.info(message_sentence)

    if len(keywords) > 0:
        message_keywords = " ".join([
            ' ' * len(prefix),
            '>' * len(boundary_indicators),
            ', '.join(keywords)
        ])

        print(message_keywords)
        segmentation_logger.info(message_keywords)

        # print(
        #     ' ' * len(prefix),
        #     '>' * len(boundary_indicators),
        #     ', '.join(keywords),
        #     sep=' '
        # )

async def dummy_transcribe_audio_stream(stream_url, step_s, model, language, max_duration, verbosity, print_openai, whisper_cpp_root_path):
    logger.info("Started audio transcribation.")
    async def generator():
        # Dummy transcribed segments with mock start and end times
        dummy_transcriptions = [
            {"text": "Hello, how are you?", "start": 0, "end": 3},
            {"text": "I am fine, thank you.", "start": 4, "end": 7},
            {"text": "What about you,", "start": 8, "end": 10},
            {"text": "Акимов Юрий?", "start":11, "end": 12},
            {"text": "I'm doing well, thanks.", "start": 12, "end": 14},
        ]

        for transcription in dummy_transcriptions:
            yield transcription

    return generator()

async def transcribe_audio_stream(stream_url, step_s, model, language, max_duration, verbosity, print_openai, whisper_cpp_root_path):
    logger.info("Started audio transcribation.")
    command = f"""
    cd {whisper_cpp_root_path}
    ./examples/livestream.sh "{stream_url}" {str(step_s)} {model} {language} {str(max_duration)} {str(verbosity)} {str(print_openai)}
    """
    logger.debug(f"Launching whisper.cpp with shell script: {command}")

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
                log_fn(line)
                yield line

        # Yield from stdout while logging stderr
        async for line in read_stream(process.stdout, log_fn=logger.debug):
            # TODO: more meaningful message for json.decoder.JSONDecodeError - WhisperCPP does not output json
            if line != "":
                transcribed_segment = json.loads(line)
                # print(transcribed_segment)
                yield transcribed_segment

        async for line in read_stream(process.stderr, log_fn=logger.error):
            pass

        # Wait for the process to finish
        await process.wait()

    # Return the constructed async generator
    return generator()

async def generate_sentences(transcript_generator):
    async def sentence_generator():
        logger.debug('Generating another sentence...')
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
                logger.debug(f'A sentence has been generated: {current_text}')

                # Yield the complete sentence
                yield {
                    "start": start_time,
                    "end": transcript["end"],
                    "text": current_text
                }

                logger.debug('Generating another sentence...')

                # Reset the buffer and start time
                sentence_buffer = []
                start_time = None

        # Handle any remaining text in the buffer after the generator ends
        if sentence_buffer:
            logger.debug(f'A sentence has been generated: {transcript["text"]}')

            yield {
                "start": start_time,
                "end": transcript["end"],
                "text": " ".join(sentence_buffer).strip()
            }

    return sentence_generator()

def dummy_load_model_from_wandb():
    class DummyPredictor:
        def predict(self, documents, pretokenized_sents=None):
            # Simulate boundary prediction for testing
            # Assuming each document is a list of sentences
            results = []
            for doc in documents:
                boundaries = [False] * (len(doc) - 1) + [True]
                results.append({"boundaries": [boundaries]})
            return results

    logger.info("Loaded dummy predictor model.")
    return DummyPredictor()

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
    logger.info('Done loading model.')

    return predictor_model

from dotenv import load_dotenv

async def main(dev_run=False):
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
    parser.add_argument("--dev_run", type=bool, default=False, help="Run in development mode.")
    parser.add_argument("--profile", type=bool, default=False, help="Measure execution time of main functions. Log to logs/profiling.log")
    
    # Parse the arguments
    args = parser.parse_args()

    if args.stream_url is None or args.stream_url == "":
        stream_url_config = "configs/stream_url.env"
        logger.info(f"Stream URL is not provided, loading from `{stream_url_config}`")
        load_dotenv(stream_url_config)
        STREAM_URL = os.getenv("STREAM_URL")
    else:
        STREAM_URL = args.stream_url

    # TODO: fix silent hang on invalid stream_url in whisper.cpp/livestream.sh
    logger.info(f"Transcribing from stream URL: {STREAM_URL}")

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

    kwargs = dict(
        stream_url=STREAM_URL,
        step_s=args.step_s,
        model=args.model,
        language=args.language,
        max_duration=args.max_duration,
        verbosity=args.verbosity,
        print_openai=args.print_openai,
        whisper_cpp_root_path=args.whisper_cpp_root_path
    )

    if args.dev_run:
        logger.info("Starting dev run.")

        predictor_model = dummy_load_model_from_wandb()

        transcript_generator = await dummy_transcribe_audio_stream(**kwargs)
        if args.profile:
            transcript_generator = timed_generator(transcript_generator)

        sentence_generator = await generate_sentences(transcript_generator)
        if args.profile:
            sentence_generator = timed_generator(sentence_generator)

        segment_generator = await generate_segments(
            sentence_generator,
            predictor_model=predictor_model,
            buffer_size=2
        )
        if args.profile:
            segment_generator = timed_generator(segment_generator)
    else:
        logger.info("Starting normal run.")
        
        predictor_model = load_model_from_wandb()

        transcript_generator = timed_generator(await transcribe_audio_stream(**kwargs))
        if args.profile:
            transcript_generator = timed_generator(transcript_generator)

        sentence_generator = await generate_sentences(transcript_generator)
        if args.profile:
            sentence_generator = timed_generator(sentence_generator)

        segment_generator = await generate_segments(
            sentence_generator,
            predictor_model=predictor_model,
            buffer_size=10
        )
        if args.profile:
            segment_generator = timed_generator(segment_generator)


    await classify_segments(segment_generator)

from functools import partial

if __name__ == "__main__":
    asyncio.run(partial(main)())