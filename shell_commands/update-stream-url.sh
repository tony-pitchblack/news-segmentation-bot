git clone https://github.com/tony-pitchblack/news-segmentation-bot
chmod -R 755 news-segmentation-bot
sudo ./news-segmentation-bot/scripts/install_selenium.sh
python3 -m venv news-bot-env
source news-bot-env/bin/activate
pip install -r news-segmentation-bot/requirements.txt
