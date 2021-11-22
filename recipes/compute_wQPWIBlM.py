# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#%config Completer.use_jedi = False

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import os
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map
import urllib.request

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
inp_eng = dataiku.Dataset("FEIDEGGER_release_1_0_english")
inp_eng_df = inp_eng.get_dataframe()

#inp_ger = dataiku.Dataset("FEIDEGGER_release_1_1_json")
#inp_ger_df = inp_ger.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
images = dataiku.Folder("wQPWIBlM")
images_info = images.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
inp_eng_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
url_list = list(inp_eng_df['Image URL'].values)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def download_image(url):
    image_path = images.file_path(url.split('/')[-1])
    if not os.path.exists(image_path):
        urllib.request.urlretrieve(url, image_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#def download_image(url):
#    urllib.request.urlretrieve(url, images.file_path(url.split('/')[-1]))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
workers = cpu_count()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from tqdm import tqdm

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for url in tqdm(url_list):
    download_image(url)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#process_map(download_image, url_list, max_workers=workers, chunksize=1)