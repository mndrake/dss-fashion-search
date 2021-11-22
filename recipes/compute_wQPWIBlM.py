# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
feidegger_release_1_0_english = dataiku.Dataset("FEIDEGGER_release_1_0_english")
feidegger_release_1_0_english_df = feidegger_release_1_0_english.get_dataframe()




# Write recipe outputs
images = dataiku.Folder("wQPWIBlM")
images_info = images.get_info()
