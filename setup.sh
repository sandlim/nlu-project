virtualenv -p python3 .env
source .env/bin/activate
pip install -r --system-site-packages requirements.txt

cd data
./download.sh
