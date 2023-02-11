"""Trains and tests a classifier of OGT.

Parameters:
- `split_type`: 'random', 'taxa_id', int
  - random: randomly splits proteins into train and dev
  - taxa_id: keeps proteins from a particular organism together
  - int: clusters taxa by taxonomy of a particular level, 1 is kingdom, 2 is phylum, etc. Note that this may have errors, as it is based on taxonomy specified in NCBI which sometimes has ambiquity
- `ogt_window`: (float, float), low and high temparature of window between OGT classes
- `balance`: bool, whether or not to balance training set
- `model`: 'protbert' or 'DeepTP', which model to start with
- `protocol`: 'head' or 'finetune'
  - head: will only train a MLP head to the base model
  - finetune: allows predictor head and base model to be backpropegated
"""
import os
from yaml import safe_load as yaml_load
from yaml import dump as yaml_dump
import pandas as pd
import numpy as np
import duckdb as ddb

from datasets import Dataset

import data_utils

import logging

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
    LOGLEVEL = getattr(logging, LOGLEVEL)
else:
    LOGLEVEL = logging.INFO
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

if __name__ == '__main__':
    # start logger
    logger = logging.getLogger(LOGNAME)
    logger.setLevel(LOGLEVEL)
    fh = logging.FileHandler(LOGFILE, mode='w')
    logger.addHandler(fh)

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)['ogt_protein_classifier_train_evaluate']
    logger.info(f"Loaded parameters: {params}")

    # get the data
    conn = ddb.connect("./data/database")
    ds = Dataset.from_sql(
        """SELECT 
            proteins.protein_int_index,
            proteins.protein_seq,
            taxa.ogt,
            taxa.taxa_index,
            taxa.taxonomy
        FROM proteins
        INNER JOIN taxa ON (proteins.taxa_index=taxa.taxa_index)
        WHERE proteins.protein_len<250
        AND taxa.ogt IS NOT NULL
        USING SAMPLE 1000""",
        config_name='test',
        cache_dir='./tmp/hf_cache',
        con=conn)
    logger.info(f"Loaded data from database, {len(ds)} total points")

    # remove no OGT label
    ds = ds.filter(lambda e: e['ogt'] != None)
    logger.info(f"Removed examples without OGT, {len(ds)} datapoints remaining")

    # determine label window
    print(params['ogt_window'])
    low = params['ogt_window'][0]
    high = params['ogt_window'][1]
    def get_label(example):
        if example['ogt']<=low:
            example['label']=0
        elif example['ogt']>=high:
            example['label']=1
        else:
            example['label']=None
        return example
    ds = ds.map(get_label)
    logger.info('Labeled examples...')
    ds = ds.filter(lambda e: e['label'] != None)
    logger.info(f'Removed examples within window, {len(ds)} datapoints remaining')
    logger.info(f'Initial dataset balance: {sum(ds["label"])/len(ds)}')

    # split the data
    # splitter = data_utils.DataSplitter(ds)
    # data_dict = splitter.split(splittype=params['split_type'])