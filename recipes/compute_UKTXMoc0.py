# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu





# Write recipe outputs
model = dataiku.Folder("UKTXMoc0")
model_info = model.get_info()
