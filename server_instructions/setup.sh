# git clone https://github.com/tony-pitchblack/news-segmentation-bot

# ensure prerequisites
sudo apt install python3 python3-pip -y
sudo apt install python3.12-venv -y

# create env & install python dependencies
python3 -m venv ~/news-bot-env
source ~/news-bot-env/bin/activate
pip install -r ~/news-segmentation-bot/requirements.txt
chmod -R 755 ~/news-segmentation-bot

# install shell dependencies
sudo ./news-segmentation-bot/scripts/install_selenium.sh
sudo ./news-segmentation-bot/scripts/install_utils.sh

# install whisper.cpp
sudo apt install cmake
sudo apt install g++
git clone https://github.com/tony-pitchblack/whisper.cpp/

# install NSE-TopicSegmenation
git clone https://github.com/tony-pitchblack/NSE-TopicSegmentation.git ~/news-segmentation-bot/nse_topic_segmentation
pip install -r ~/news-segmentation-bot/nse_topic_segmentation/requirements.txt
cd ~/news-segmentation-bot/nse_topic_segmentation/ && git checkout continuous-model-inference

# install other dependencies
python -m spacy download ru_core_news_sm

# set local time
sudo timedatectl set-timezone Europe/Moscow