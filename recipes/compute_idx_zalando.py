# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#%config IPCompleter.use_jedi = False

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import inference

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Read recipe inputs
image_details_by_image_path = dataiku.Dataset("image_details_by_image_path")
image_details_by_image_path_df = image_details_by_image_path.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

idx_zalando_df = image_details_by_image_path_df # For this sample code, simply copy input to output


# Write recipe outputs
idx_zalando = dataiku.Dataset("idx_zalando")
idx_zalando.write_with_schema(idx_zalando_df)