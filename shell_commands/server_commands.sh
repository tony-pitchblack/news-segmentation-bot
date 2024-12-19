ssh -i C:\Users\User\.ssh\ano_default_key -l admin 158.160.141.25
apt install python3.12-venv
python3 -m venv whisper-live-env
pip install faster-whisper

cd WhisperLive/
source whisper-live-env/bin/activate
python3 run_server.py --port 9090 --backend faster_whisper -fw models/

# no sudo for docker
grep docker /etc/group
sudo groupadd docker
sudo usermod -aG docker admin
newgrp docker

# faster-whisper-server
# INSTALL
sudo apt install docker-buildx
git clone https://github.com/fedirz/faster-whisper-server.git
git clone https://github.com/tony-pitchblack/faster-whisper-server
cd faster-whisper-server
git checkout truncated-confirmed-sending

# GIT RESOLVE CONFLICTS POLICY
git config --global pull.rebase true

# BUILD
cd faster-whisper-server/ && \
git pull origin master && \
git checkout master && \
sudo docker build -t faster-whisper-server:local -f Dockerfile.cpu . --rm \
&& cd ..

# BUILD
cd faster-whisper-server/ && \
git pull origin truncated-confirmed-sending && \
git checkout truncated-confirmed-sending && \
sudo docker build -t faster-whisper-server:local-truncated -f Dockerfile.cpu . --rm \
&& cd ..

# RUN
sudo docker run \
  --publish 8000:8000 \
  --volume ~/.cache/huggingface:/root/.cache/huggingface \
  --env-file .env \
  faster-whisper-server:local
  # faster-whisper-server:local-truncated
  # faster-whisper-server:websocket-debug

# ALIAS
sudo docker kill $(sudo docker ps -q)
alias kill_docker='sudo docker kill $(sudo docker ps -q)'
source ~/.bashrc

# CLEAR INTERMEDIATE FILES
docker container prune
docker image prune -a

# DELETE ALL
docker system prune -af

# EDIT FILES
vim ~/faster-whisper-server/src/faster_whisper_server/transcriber.py
vim ~/faster-whisper-server/src/faster_whisper_server/routers/stt.py