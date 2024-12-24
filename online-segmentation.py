import json
import numpy as np
from datetime import timedelta

def sentence_generator(word_objects, sentence_buffer=None):
    """
    Generates sentences from a stream of word objects, maintaining a reusable buffer.

    Each word object should have 'word', 'start', and 'end' keys.
    A sentence is returned when a word ends with a sentence-ending punctuation
    (e.g., ".", "?", "!"). The function also returns the updated buffer.

    Args:
        word_objects (list): List of word objects (e.g., [{"word": "Hello", "start": 0.0, "end": 0.5}]).
        sentence_buffer (list): Optional. A list of previously buffered word objects.

    Returns:
        list of tuples: Each tuple contains a sentence (str), start time (float), and end time (float).
        list: The updated sentence buffer.
    """
    if sentence_buffer is None:
        sentence_buffer = []

    sentences = []
    sentence_buffer.extend(word_objects)

    current_sentence = []
    start_time = None

    for word_obj in sentence_buffer:
        word = word_obj['word']
        start = word_obj['start']
        end = word_obj['end']

        if not current_sentence:
            start_time = start

        current_sentence.append(word)

        # Check if this word ends the sentence
        if word.endswith(('.', '?', '!')):
            sentence_text = ' '.join(current_sentence)
            # sentences.append((sentence_text, start_time, end))
            sentences.append({"text": sentence_text, "start": start_time, "end": end})
            current_sentence = []
            start_time = None

    # Update the buffer to include only the unprocessed words
    if current_sentence:
        sentence_buffer = [{'word': w, 'start': start_time, 'end': word_obj['end']} for w in current_sentence]
    else:
        sentence_buffer = []

    return sentences, sentence_buffer

async def receive_single_message(
    websocket, predictor_model, ws_timeout,
    total_word_count, sentence_buffer, prev_start,
    # proc_id
    ):

    """Asynchronously receive and process messages from the WebSocket."""

    response = await asyncio.wait_for(websocket.recv(), timeout=ws_timeout)
    # print(f"Received response. ID: {proc_id}, size: {len(response)}.")

    # Parse JSON response
    response_data = json.loads(response)

    # Extract words and filter by latest timestamp
    word_objects = response_data.get('words', [])
    filtered_words = word_objects[total_word_count:]

    if filtered_words:
        total_word_count = len(word_objects)

    # Generate sentences from filtered words
    sentences, sentence_buffer = sentence_generator(filtered_words, sentence_buffer)

    # print()
    # print(f'response_data: {response_data}')
    # print(f'received words: {word_objects}')
    # print(f'filtered_words: {filtered_words}')
    # print(f'sentence_buffer: {sentence_buffer}')

    # If there are sentences, prepare them for segmentation inference
    if len(sentences) > 0:
        sentences_texts = [sentence['text'] for sentence in sentences]
        doc = [sentences_texts]

        # Predict boundaries using the text segmentation model
        predictions = predictor_model.predict([doc], pretokenized_sents=doc)
        boundary_mask = predictions[0]['boundaries'][0]
        boundary_mask = np.array(boundary_mask)

        # Get indices of sentences marked as boundaries
        boundary_idx = np.where(boundary_mask)[0]

        # Process each sentence and boundary prediction
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

    return total_word_count, sentence_buffer, prev_start

async def receive_messages(websocket, predictor_model, ws_timeout):
    """Asynchronously receive and process messages from the WebSocket."""
    total_word_count = 0
    sentence_buffer = []  # Buffer for assembling sentences
    prev_start = 0  # Tracks the previous sentence's start time for fallback

    # proc_id = 0
    while True:
        try:
            total_word_count, sentence_buffer, prev_start = await receive_single_message(
                websocket, predictor_model, ws_timeout,
                total_word_count, sentence_buffer, prev_start,
                # proc_id
            )
            # proc_id += 1
        except asyncio.TimeoutError:
            print("receive_messages: WebSocket receive timeout. No message received.")
            continue
        except websockets.ConnectionClosed as e:
            print(f"receive_messages: WebSocket connection closed: {e}")
            break
        except Exception as e:
            print(f"receive_messages: Error processing WebSocket message: {e}")
            raise

import asyncio
import websockets

async def send_audio_chunks(process1_stdout, websocket):
    """Asynchronously send audio chunks to the WebSocket."""
    try:
        while True:
            try:
                chunk = await process1_stdout.readexactly(1024)  # Read 1024-byte chunks
                await websocket.send(chunk)
            except asyncio.IncompleteReadError as e:
                # Handle incomplete read (EOF or fewer bytes than expected)
                if e.partial:
                    await websocket.send(e.partial)  # Send remaining bytes, if any
                print("End of stream.")
                break
    except websockets.ConnectionClosed as e:
        print(f"send_audio_chunks: WebSocket connection closed while sending: {e}")
    except Exception as e:
        print(f"send_audio_chunks: Error while sending audio chunks: {e}")

# Function to format time in "hh:mm:ss.xx"
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs % 1) * 100)
    secs = int(secs)  # Remove fractional part for formatting
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:02}"

async def rebroadcast_audio(stream_url, whisper_server_url, trim_seconds=20, recv_msg_kwargs={}):
    # First subprocess: ffmpeg processes the m3u8 stream
    trim_option = f"-t {trim_seconds}" if trim_seconds is not None else ''
    process1_command = f"""
        ffmpeg -i {stream_url} -loglevel debug {trim_option} -vn -acodec copy -f wav pipe:1 | \
        ffmpeg -y -i pipe:0 -ac 1 -ar 16000 -f s16le -acodec pcm_s16le pipe:1 | \
        pv -qL 32000
    """

    # Launch the first ffmpeg process
    process1 = await asyncio.create_subprocess_shell(
        process1_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async with websockets.connect(whisper_server_url, ping_interval=None) as websocket:
        try:
            # Start sending audio chunks and receiving messages concurrently
            send_task = asyncio.create_task(send_audio_chunks(process1.stdout, websocket))
            receive_task = asyncio.create_task(receive_messages(websocket, **recv_msg_kwargs))

            # Wait for process1 to finish
            await process1.wait()
            # print(f"Process1 exited with code {process1.returncode}")

            # Wait for both send and receive tasks to finish
            await asyncio.gather(send_task, receive_task)

        except Exception as e:
            print(f"Error during WebSocket handling: {e}")
            # send_task.cancel()
            # receive_task.cancel()
            # await asyncio.gather(send_task, receive_task)
            raise

        # finally:
        #     if process1.returncode is None:  # Ensure ffmpeg process is terminated
        #         process1.terminate()

from utils import get_whisper_server_url
from dotenv import load_dotenv
import os

# import nltk
# nltk.download('punkt_tab')

def load_model_from_wandb(run_id='k4j7vuo7'):
    api = wandb.Api()
    artifact = api.artifact(f'overfit1010/lenta_BiLSTM_F1/model-{run_id}:v0', type='model')
    art_dir = artifact.download()
    ckpt_path = os.path.join(art_dir, 'model.ckpt')

    text_seg_model = TextSegmenter.load_from_checkpoint(ckpt_path).to('cpu')
    predictor_model = Predictor(text_seg_model, sentence_encoder="cointegrated/rubert-tiny2")

    return predictor_model

if __name__ == '__main__':
    # Load env variables
    load_dotenv("configs/general.env")
    NEWS_URL = os.getenv("NEWS_URL")

    # Load env variables
    load_dotenv("configs/websocket.env")
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")
    MODEL = os.getenv("MODEL")
    
    whisper_server_url = get_whisper_server_url(HOST, MODEL)
    predictor_model = load_model_from_wandb()

    # Measure exec time if requested
    USE_MEASURE_EXECUTION_TIME = False
    funcs_to_measure_exec_time = [
        'send_audio_chunks',
        'receive_messages',
        'receive_single_message',
        'rebroadcast_audio'
    ]

    if  USE_MEASURE_EXECUTION_TIME:
        for func in funcs_to_measure_exec_time:
            if func in globals():
                globals()[func] = measure_execution_time(globals()[func])


    # Run the rebroadcast
    await rebroadcast_audio(
        # stream_url=DUMMY_STREAM_URL,
        stream_url=NEWS_URL,

        trim_seconds=int(60 * 2),
        # trim_seconds=30,

        whisper_server_url=whisper_server_url,
        recv_msg_kwargs = dict(
                predictor_model=predictor_model,
                ws_timeout=int(60 * 5)
                # ws_timeout=60
            )
        )

        whisper_server_url = get_whisper_server_url(
            host=HOST,
            port=PORT,

            # model = "Systran/faster-distil-whisper-large-v3",
            # model = "Systran/faster-whisper-large-v3",
            # model = "Systran/faster-whisper-medium",
            # model = "Systran/faster-whisper-base",
            # model = "Systran/faster-whisper-small",
            model = "Systran/faster-whisper-tiny",

            language='ru',
            # language='en',
        )