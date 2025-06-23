import sys
import os
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import DependencyManager
deps = DependencyManager()

kaggle = deps.auto_import('kaggle')

kaggle_dir = os.path.expanduser('~/.kaggle')
kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

if not os.path.exists(kaggle_json):
    raise FileNotFoundError(f'kaggle.json não encontrado: {kaggle_json}')

try:
    os.chmod(kaggle_json, 0o600)
except Exception:
    pass

if not os.path.exists(kaggle_json):
    raise FileNotFoundError(f'kaggle.json nao encontrado: {kaggle_json}')

download_command = 'kaggle datasets download -d mlg-ulb/creditcardfraud'
ret_code = os.system(download_command)

if ret_code != 0:
    raise RuntimeError(f'erro em: {download_command}')

with zipfile.ZipFile('creditcardfraud.zip', 'r') as zip_ref:
    zip_ref.extractall('creditcard_fraud_dataset')

print('dataset baixado e extraido')
