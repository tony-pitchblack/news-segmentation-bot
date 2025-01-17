#!/bin/bash

# Default values
language="ru"
model="small"
trim_duration=${1:-60}

# Update stream url
source ~/news-bot-env/bin/activate
python3 ~/news-segmentation-bot/update-stream-url.py --source ntv # update STREAM_URL
deactivate

# Load stream url
source ~/news-segmentation-bot/configs/stream_url.env

# Download model
cd ~/whisper.cpp
./models/download-ggml-model.sh $model

# Download and trim the video
ffmpeg -y -hide_banner -loglevel quiet -i "$STREAM_URL" -t $trim_duration -ac 1 -ar 16000 -acodec pcm_s16le /tmp/whisper-live.wav
ffprobe /tmp/whisper-live.wav

# Debug whisper-cli
time ./build/bin/whisper-cli \
  -t 12 \
  -m ./models/ggml-${model}.bin \
  -f /tmp/whisper-live.wav \
  --language $language \
  --no-timestamps \
  # 2> /tmp/whispererr