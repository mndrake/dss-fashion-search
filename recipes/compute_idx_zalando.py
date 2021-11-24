# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#%config IPCompleter.use_jedi = False

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu 
import inference
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_path = dataiku.Folder("UKTXMoc0").get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model = inference.model_fn(model_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
image_details_by_image_path = dataiku.Dataset("image_details_by_image_path")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
es = Elasticsearch()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def make_feature_vec(row, model):
    data = row['description_concat']
    feature_vec = inference.predict_fn(data, model)
    return feature_vec

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_dim = len(inference.predict_fn("blue dress", model))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# create elasticsearch Index
if es.indices.exists(index='idx_zalando'):
    es.indices.delete(index='idx_zalando')
    
create_query = {
    "mappings": {
        "properties": {
            "zalando_nlu_vector": {
                "type": "dense_vector",
                "dims": model_dim
            }
        }
    }
}

es.indices.create(index='idx_zalando', **create_query)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_embeddings():
    for row in image_details_by_image_path.iter_rows():
        embeddings = make_feature_vec(row, model)
        row['zalando_nlu_vector'] = embeddings
        yield row

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
bulk(es, get_embeddings())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# # Read recipe inputs
# image_details_by_image_path = dataiku.Dataset("image_details_by_image_path")
# image_details_by_image_path_df = image_details_by_image_path.get_dataframe()


# # Compute recipe outputs from inputs
# # TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# # NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

# idx_zalando_df = image_details_by_image_path_df # For this sample code, simply copy input to output


# # Write recipe outputs
# idx_zalando = dataiku.Dataset("idx_zalando")
# idx_zalando.write_with_schema(idx_zalando_df)