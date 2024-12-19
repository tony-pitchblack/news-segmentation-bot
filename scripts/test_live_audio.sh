cd ~/faster-whisper-server/examples/live-audio
cat audio.pcm | pv -qL 32000 | websocat --no-close --binary 'ws://localhost:8000/v1/audio/transcriptions?response_format=verbose_json&language=en'
cd ~