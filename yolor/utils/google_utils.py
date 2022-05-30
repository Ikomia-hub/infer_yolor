# Google utils: https://cloud.google.com/storage/docs/reference/libraries
import os
# import platform
import subprocess
# import time
from pathlib import Path
import requests
import torch


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output('gsutil du %s' % url, shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    file = Path(weights).name

    msg = weights + ' missing, try downloading from https://github.com/WongKinYiu/yolor/releases/'
    models = ['yolor_p6.pt', 'yolor_w6.pt']  # available models

    if file in models and not os.path.isfile(weights):

        try:  # GitHub
            url = 'https://github.com/WongKinYiu/yolor/releases/download/v1.0/' + file
            print('Downloading %s to %s...' % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('ERROR: Download failure.')
            print('')
            
            
def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


# def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
#     # Downloads a file from Google Drive. from utils.google_utils import *; gdrive_download()
#     t = time.time()
#
#     print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
#     os.remove(name) if os.path.exists(name) else None  # remove existing
#     os.remove('cookie') if os.path.exists('cookie') else None
#
#     # Attempt file download
#     out = "NUL" if platform.system() == "Windows" else "/dev/null"
#     os.system('curl -L -c cookie "https://docs.google.com/uc?export=download&id=%s" | sed -rn "s/.*confirm=(['
#               '0-9A-Za-z_]+).*/\1/p" > confirm.txt' % (id))
#     if os.path.exists('cookie'):  # large file
#         s = 'curl -L -b cookie -o %s "https://docs.google.com/uc?export=download&id=%s&confirm=%s"' % (name, id, 'confirm.txt')
#     else:  # small file
#         s = 'curl -s -L -o %s "https://docs.google.com/uc?export=download&id=%s"' % (name, id)
#     r = os.system(s)  # execute, capture return
#     os.remove('cookie') if os.path.exists('cookie') else None
#     os.remove('confirm.txt') if os.path.exists('confirm.txt') else None
#     # Error check
#     if r != 0:
#         os.remove(name) if os.path.exists(name) else None  # remove partial
#         print('Download error ')  # raise Exception('Download error')
#         return r
#
#     # Unzip if archive
#     if name.endswith('.zip'):
#         print('unzipping... ', end='')
#         os.system('unzip -q %s' % name)  # unzip
#         os.remove(name)  # remove zip to free space
#
#     print('Done (%.1fs)' % (time.time() - t))
#     return r


def gdrive_download(file_id, dst_path):
    path = os.path.join(os.path.dirname(__file__), "download_model")
    if not os.path.exists(path):
        os.makedirs(path)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': file_id, "confirm": "t"}, stream=True)
    save_response_content(response, dst_path)


def save_response_content(response, dst_path):
    CHUNK_SIZE = 32768

    with open(dst_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""

# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
