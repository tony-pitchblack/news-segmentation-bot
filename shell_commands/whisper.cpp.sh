#!/bin/bash
git clone https://github.com/tony-pitchblack/whisper.cpp.git
# cd whisper.cpp && ./models/download-ggml-model.sh $model && cd ~
./models/download-ggml-model.sh $model

# Download and trim the video
ffmpeg -y -hide_banner -loglevel quiet -i "$STREAM_URL" -t 60 -ac 1 -ar 16000 -acodec pcm_s16le /tmp/whisper-live.wav

# Debug whisper-cli
cd whisper.cpp
source ~/whisper.cpp.env
time ./build/bin/whisper-cli \
  -t 8 \
  -m ./models/ggml-${model}.bin \
  -f /tmp/whisper-live.wav \
  --language $language \
  --no-timestamps \
  -otxt 2> /tmp/whispererr

time ./build/bin/whisper-cli \
  -t 8 \
  -m ./models/ggml-${model}.bin \
  -f /tmp/whisper-live.wav \
  --language $language \
  -poai 2> /tmp/whispererr

# Debug whisper-cli.cpp
cd whisper.cpp
source ~/whisper.cpp.env
time ./examples/livestream.sh $STREAM_URL 15 small ru 60
time ./examples/livestream.sh $STREAM_URL 15 small ru 60 0 1

# Debug whisper_streaming.py
cd whisper.cpp
source ~/whisper.cpp.env
python3 ./examples/python/whisper_streaming.py $STREAM_URL --step_s 15 --model small --language ru --max_duration 60 --verbosity 0 --print_openai 1
python3 ./examples/python/whisper_streaming_standalone.py $STREAM_URL --step_s 15 --model small --language ru --max_duration 60 --verbosity 0 --print_openai 1

# Debug whisper-cpp-client.py
python3 whisper-cpp-client.py $STREAM_URL --step_s 15 --model small --language ru --max_duration 60 --verbosity 0 --print_openai 1
