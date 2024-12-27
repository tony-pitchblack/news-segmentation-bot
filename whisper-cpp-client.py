import asyncio
import sys
import os
import json

async def run_livestream(stream_url, step_s, model, language, max_duration, verbosity, print_openai, whisper_cpp_root_path):
    """Run the livestream.sh script and process its output."""

    command = f"""
    cd {whisper_cpp_root_path}
    ./examples/livestream.sh "{stream_url}" {str(step_s)} {model} {language} {str(max_duration)} {str(verbosity)} {str(print_openai)}
    """

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def read_stream(stream, log_prefix):
        """Read and load JSON from a stream, then print it."""
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                line_decoded = line.decode().strip()
                if line_decoded != "": # ignore extra newlines between chunks of segments
                    data = json.loads(line_decoded)
                    print(f"{log_prefix}{json.dumps(data, indent=2, ensure_ascii=False)}")  # Pretty print JSON

                # data = json.loads(line.decode().strip())
                # print(f"{log_prefix}{json.dumps(data, indent=2)}")  # Pretty print JSON
            except json.JSONDecodeError:
                print(f"{log_prefix}Error: Invalid JSON on line: {line.decode().strip()}")

    # Run stdout and stderr readers concurrently
    stdout_task = asyncio.create_task(read_stream(process.stdout, ""))
    stderr_task = asyncio.create_task(read_stream(process.stderr, "[stderr] "))

    await asyncio.wait([stdout_task, stderr_task])
    return await process.wait()

from nse_topic_segmentation.models.lightning_model import TextSegmenter
from nse_topic_segmentation.models.EncoderDataset import Predictor
import wandb

def load_model_from_wandb(run_id='k4j7vuo7'):
    api = wandb.Api()
    artifact = api.artifact(f'overfit1010/lenta_BiLSTM_F1/model-{run_id}:v0', type='model')
    art_dir = artifact.download()
    ckpt_path = os.path.join(art_dir, 'model.ckpt')

    text_seg_model = TextSegmenter.load_from_checkpoint(ckpt_path).to('cpu')
    predictor_model = Predictor(text_seg_model, sentence_encoder="cointegrated/rubert-tiny2")


from dotenv import load_dotenv

def main():
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run the livestream with the specified parameters.")
    
    # Define required positional argument for the stream URL
    parser.add_argument("stream_url", type=str, default=None, help="The URL of the stream.")
    
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
        print("Stream URL is not provided, loading STREAM_URL from `configs/news_url.env` ...")
        load_dotenv("configs/news_url.env")
        STREAM_URL = os.getenv("STREAM_URL")

    predictor_model = load_model_from_wandb()

    # Run the asyncio event loop
    asyncio.run(
        run_livestream(
            STREAM_URL, args.step_s, args.model, 
            args.language, args.max_duration, 
            args.verbosity, args.print_openai, 
            args.whisper_cpp_root_path
        )
    )


if __name__ == "__main__":
    main()