echo "STEP 1: unzip data"
tar --no-same-owner --no-same-permissions -xzf imagenet100_128x128.tar.gz
mkdir -p data
mv imagenet100_128x128 data/

echo "STEP 2: pip install"
pip install -r requirements.txt
