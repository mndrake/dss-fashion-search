# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
model_folder = dataiku.Folder("UKTXMoc0")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tokenizer.save_pretrained(model_folder.get_path())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model.save_pretrained(model_folder.get_path())