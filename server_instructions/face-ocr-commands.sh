git clone https://github.com/tony-pitchblack/face-recognition-ocr-bot
python3.10 -m venv venv
source venv/bin/activate
pip install requirements.txt
chmod +x ./scripts/test_pipeline.sh
./scripts/test_pipeline.sh
sudo apt-get install pv