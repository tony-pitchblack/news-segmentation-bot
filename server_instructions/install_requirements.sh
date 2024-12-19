# git clone https://github.com/tony-pitchblack/news-segmentation-bot

# create env & install python dependencies
python3 -m venv news-bot-env
source news-bot-env/bin/activate
pip install -r news-segmentation-bot/requirements.txt

# install shell dependencies
chmod -R 755 news-segmentation-bot
sudo ./news-segmentation-bot/scripts/install_selenium.sh
sudo ./news-segmentation-bot/scripts/install_utils.sh

# install topic-segmentation dependencies
sudo ./news-segmentation-bot/scripts/install_utils.sh
git clone https://github.com/tony-pitchblack/NSE-TopicSegmentation.git
pip install -r NSE-TopicSegmentation/requirements.txt