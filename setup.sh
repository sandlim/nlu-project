virtualenv --system-site-packages -p python3 .env
source .env/bin/activate
pip install -r requirements.txt

cd data
./download.sh
