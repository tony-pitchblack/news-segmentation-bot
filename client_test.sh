# test hello world
curl -k "https://very-unique-sub-domain.loca.lt/hello"

# test setting ws url
curl -k "https://very-unique-sub-domain.loca.lt/set-websocket-url?url=ws://158.160.144.180:8000/v1/audio/transcriptions?language=ru&model=Systran/faster-whisper-tiny&response_format=verbose_json"

# test upload-audio
curl -k "https://very-unique-sub-domain.loca.lt/upload-audio?url=ws%3A%2F%2F158.160.144.180%3A8000%2Fv1%2Faudio%2Ftranscriptions%3Flanguage%3Dru%26model%3DSystran%2Ffaster-whisper-tiny%26response_format%3Dverbose_json&filename=audio.pcm"