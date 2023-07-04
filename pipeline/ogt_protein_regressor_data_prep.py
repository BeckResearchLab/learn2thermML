"""Trains and tests a regressor of OGT.

Parameters:
- `split_type`: 'random', 'taxaid', int
  - random: randomly splits proteins into train and dev
  - taxa_id: keeps proteins from a particular organism together
  - int: clusters taxa by taxonomy of a particular level, 1 is kingdom, 2 is phylum, etc. Note that this may have errors, as it is based on taxonomy specified in NCBI which sometimes has ambiquity
- `max/min_protein_len`: int, maximum and minumum protein length to consider
- `deduplication`: 
   - `do`: bool, whether to do deduplication
   - `num_perm`: int, number of MinHash permutations
   - `kgram`: int, size of kgrams for protein representation
   - `jaccard`: float remove examples within the dataset with > the value in jaccard distance based on MinHash of k-gram protein representation. 
- `data_batch_size`: int, batch size for data processing steps
- `dev_keep_columns`: bool, keep datra columns not needed for ML in the HF dataset
- `dev_sample_init_data`: bool, work with a small test sample or not
"""
import os
import time
from yaml import safe_load as yaml_load
from yaml import dump as yaml_dump
import json
import pandas as pd
import numpy as np
import duckdb as ddb
import re

import datasets
import sklearn.utils
import codecarbon

if 'SLURM_NTASKS' in os.environ:
    CPU_COUNT = int(os.environ['SLURM_NTASKS'])
else:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()

import l2tml_utils.data_utils as data_utils

import logging
logger = logging.getLogger(__name__)

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

if __name__ == '__main__':
    # start logger
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    utils_logger = logging.getLogger('l2tml_utils')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)
    #datasets_logger = logging.getLogger('datasets')
    #datasets_logger.setLevel(getattr(logging, LOGLEVEL))
    #datasets_logger.addHandler(fh)
    
    # create dirs
    if not os.path.exists('./data/ogt_protein_regressor/data'):
        os.makedirs('./data/ogt_protein_regressor/data')

    # device settings
    logger.info(f'Using {CPU_COUNT} cpus for multiprocessing')

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)['ogt_protein_regressor_data_prep']
    logger.info(f"Loaded parameters: {params}")
    dedup_dict = {}
    for d in params['deduplication']:
        dedup_dict.update(d)
    params['deduplication'] = dedup_dict
    balance_dict = {}
    for d in params['balancing']:
        balance_dict.update(d)
    params['balancing'] = balance_dict
    
    # set some kwargs for datasets batching
    ds_batch_params = dict(batched=True, batch_size=params['data_batch_size'], num_proc=CPU_COUNT)
    
    # start carbon tracker for data processing
    data_tracker = codecarbon.OfflineEmissionsTracker( 
        project_name="data_process_regressor",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
    )
    data_tracker.start()

    # get the data
    conn = ddb.connect("./data/database")
    # create indexes first
    select_statement = f"""SELECT 
            proteins.pid,
            proteins.protein_seq,
            taxa.temperature,
            taxa.taxid,
        FROM proteins
        INNER JOIN taxa ON (proteins.taxid=taxa.taxid)
        WHERE LEN(proteins.protein_seq)<={params['max_protein_len']}
        AND LEN(proteins.protein_seq)>{params['min_protein_len']}
        AND taxa.temperature IS NOT NULL"""
    if params['dev_sample_init_data']:
        select_statement = select_statement + f" USING SAMPLE {params['dev_sample_init_data']}"
    ds = datasets.Dataset.from_sql(
        select_statement,
        config_name='test',
        cache_dir='./tmp/hf_cache',
        con=conn)
    logger.info(f"Loaded data from database, {len(ds)} total points")
    conn.close()

    # balance based on OGT
    if params['balancing']['do']:
        ds, og_bin_sizes, new_bin_sizes, bin_edges = data_utils.regression_bin_undersampling(
            dataset=ds,
            label='temperature',
            num_bins=params['balancing']['num_bins'],
            max_bin_size=params['balancing']['max_bin_size'],
            batch_size=ds_batch_params['batch_size'],
            num_proc=ds_batch_params['num_proc'],
            min_total_data_kept=params['balancing']['min_total_data_kept'],
        )
    
    # deduplication
    if params['deduplication']['do']:
        logger.info("Deduplicating dataset...")
        j_thresh = params['deduplication']['jaccard']
        if not (j_thresh > 0.0 and j_thresh < 1.0):
            raise ValueError('Jaccard threshold for deduplication must be in [0.0,1.0]')
        import l2tml_utils.dataset_deduplication
        ds, _ = l2tml_utils.dataset_deduplication.deduplicate_dataset(
            ds,
            jaccard_threshold=j_thresh,
            num_perm=params['deduplication']['num_perm'],
            k=params['deduplication']['kgram']
        )
    else:
        pass
    logger.info(f"Dataset has {len(ds)} points after deduplication.")
    # rename OG
    ds = ds.rename_column('temperature', 'labels')
    
    # split the data
    ds = ds.shuffle()
    splitter = data_utils.DataSplitter(ds)
    data_dict = splitter.split(splittype=params['split_type'], frac=params['train_test_frac'])
    data_dict = data_dict.shuffle()
    logger.info(f"Split data into train and test.")

    # compute some statistics
    # take a small sample
    if len(data_dict['train']) > 5000:
        train_sample = data_dict['train'].select(range(1000))
    else:
        train_sample = data_dict['train']
    if len(data_dict['test']) > 5000:
        test_sample = data_dict['test'].select(range(1000))
    else:    
        test_sample = data_dict['test']
    import matplotlib.pyplot as plt
    import seaborn as sns
    if not os.path.exists('./data/ogt_protein_regressor/data_plots/'):
        os.makedirs('./data/ogt_protein_regressor/data_plots/')
    fig, ax = plt.subplots()
    sns.kdeplot(train_sample['labels'], label="train", ax=ax)
    sns.kdeplot(test_sample['labels'], label="test", ax=ax)
    plt.legend()
    plt.xlabel("OGT")
    plt.savefig('./data/ogt_protein_regressor/data_plots/ogt_kde.png', dpi=300, bbox_inches='tight')

    train_mean, train_std = float(np.mean(data_dict['train']['labels'])), float(np.std(data_dict['train']['labels']))
    test_mean, test_std = float(np.mean(data_dict['test']['labels'])), float(np.std(data_dict['test']['labels']))
    logger.info(f"Train OGT mean, std: {(train_mean, train_std)}")
    logger.info(f"Test OGT mean, std: {(test_mean, test_std)}")

    # standardize the label
    def standardize_labels(examples):
        examples['labels'] = (examples['labels'] - train_mean) / train_std
        return examples
    data_dict = data_dict.map(standardize_labels, desc='Standardizing data', batched=False)
    # save the standardization parameters to file
    with open('./data/ogt_protein_regressor/data/standardization_params.json', 'w') as f:
        json.dump({'train_mean': train_mean, 'train_std': train_std}, f)

    # remove unnecessary columns
    if not params['dev_keep_columns']:
        bad_columns = [k for k in data_dict['train'].column_names if k not in['protein_seq', 'labels']]
        data_dict = data_dict.remove_columns(bad_columns)

    logger.info(f'Final datasets: {data_dict}')
    data_dict.cleanup_cache_files()
    data_dict.save_to_disk('./data/ogt_protein_regressor/data/')
    logger.info("Saved data to disk.")
    
    # get co2
    co2 = data_tracker.stop()

    metrics = {
        'ogt_rgr_data_co2': float(co2),
        'ogt_rgr_data_n_train': len(data_dict['train']),
        'ogt_rgr_data_n_test': len(data_dict['test']), 
        'ogt_rgr_train_mean': float(train_mean),
        'ogt_rgr_train_std': float(train_std),
        'ogt_rgr_test_mean': float(test_mean),
        'ogt_rgr_test_mean': float(train_std)}

    # save metrics
    with open('./data/ogt_protein_regressor/data_metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
