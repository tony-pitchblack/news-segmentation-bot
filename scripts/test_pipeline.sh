# Load the environment variables from both .env files
SCRIPT_DIR=$(dirname "$0")
source $SCRIPT_DIR/../configs/general.env
source $SCRIPT_DIR/../configs/websocket.env

ffmpeg -i $NEWS_URL -loglevel debug -vn -acodec copy -f wav pipe:1 2>ffmpeg.log | \
ffmpeg -y -i pipe:0 -ac 1 -ar 16000 -f s16le -acodec pcm_s16le pipe:1 2>>ffmpeg.log | \
pv -qL 32000 | websocat -v --no-close --binary $WEBSOCKET_URL

# # Test on 'audio.pcm' in current folder
cat audio.pcm | pv -qL 32000 | websocat -v --no-close --binary 'ws://158.160.130.141:8000/v1/audio/transcriptions?model=Systran%2Ffaster-whisper-tiny&response_format=verbose_json&language=ru'