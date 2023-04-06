"""Trains and tests a regressor of OGT.

Parameters:
- `split_type`: 'random', 'taxa_id', int
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
import pandas as pd
import numpy as np
import duckdb as ddb
import re

import datasets
import sklearn.utils
import codecarbon

if 'SLURM_CPUS_ON_NODE' in os.environ:
    CPU_COUNT = int(os.environ['SLURM_CPUS_ON_NODE'])
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
        os.mkdir('./data/ogt_protein_regressor/data')

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
    data_tracker = codecarbon.EmissionsTracker( 
        project_name="data_process",
        output_dir="./data/ogt_protein_regressor/data",
    )
    data_tracker.start()

    # get the data
    conn = ddb.connect("./data/database")
    # create indexes first
    conn.execute("CREATE UNIQUE INDEX taxa_index ON taxa (taxa_index)")
    conn.commit()
    conn.execute("CREATE INDEX taxa_index_foreign ON proteins (taxa_index)")
    conn.commit()
    select_statement = f"""SELECT 
            proteins.protein_int_index,
            proteins.protein_seq,
            taxa.ogt,
            taxa.taxa_index,
            taxa.taxonomy
        FROM proteins
        INNER JOIN taxa ON (proteins.taxa_index=taxa.taxa_index)
        WHERE proteins.protein_len<{params['max_protein_len']}
        AND proteins.protein_len>{params['min_protein_len']}
        AND taxa.ogt IS NOT NULL"""
    if params['dev_sample_init_data']:
        select_statement = select_statement + " USING SAMPLE 100000"
    ds = datasets.Dataset.from_sql(
        select_statement,
        config_name='test',
        cache_dir='./tmp/hf_cache',
        con=conn)
    logger.info(f"Loaded data from database, {len(ds)} total points")
    conn.close()

    # check of we want to do a run with sample of data
    if params['dev_sample_init_data']:
        ds = ds.shuffle().select(range(10000))
        logger.info('Running with a sample of the data')
    else:
        pass

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

    # balance based on OGT
    if params['balancing']['do']:
        ds, og_bin_sizes, new_bin_sizes, bin_edges = data_utils.regression_bin_undersampling(
            dataset=ds,
            label_column='ogt',
            num_bins=params['balancing']['num_bins'],
            max_bin_size=params['balancing']['max_bin_size'],
            batch_size=ds_batch_params['batch_size'],
            num_proc=ds_batch_params['num_proc'],
        )

    # rename OGT
    def rename_ogt(examples):
        examples['labels'] = examples['ogt']
    ds = ds.map(rename_ogt, **ds_batch_params, desc="Renaming OGT to labels...")
    
    # split the data
    splitter = data_utils.DataSplitter(ds)
    data_dict = splitter.split(splittype=params['split_type'], frac=params['train_test_frac'])
    data_dict = data_dict.shuffle()
    logger.info(f"Split data into train and test.")

    # compute some statistics
    # take a small sample
    train_sample = data_dict['train'].select(range(1000))
    test_sample = data_dict['test'].select(range(1000))
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    sns.kdeplot(train_sample['ogt'], label="train", ax=ax)
    sns.kdeplot(test_sample['ogt'], label="test", ax=ax)
    plt.savefig('./data/ogt_protein_regressor/data_plots/ogt_kde.png', dpi=300, bbox_inches='tight')

    train_mean, train_std = np.mean(train_sample['ogt']), np.std(train_sample['ogt'])
    test_mean, test_std = np.mean(test_sample['ogt']), np.std(test_sample['ogt'])
    logger.info(f"Train apx OGT mean, std: {(train_mean, train_std)}")
    logger.info(f"Test apx OGT mean, std: {(test_mean, test_std)}")
    
    # remove unnecessary columns
    if not params['dev_keep_columns']:
        data_dict = data_dict.remove_columns(['protein_int_index', 'ogt', 'taxa_index', 'taxonomy'])
    logger.info(f'Final datasets: {data_dict}')
    data_dict.cleanup_cache_files()
    data_dict.save_to_disk('./data/ogt_protein_regressor/data/')
    logger.info("Saved data to disk.")
    
    # get co2
    co2 = data_tracker.stop()

    metrics = {'ogt_rgr_data_co2': co2, 'ogt_rgr_data_n_train': len(data_dict['train']), 'ogt_rgr_data_n_test': len(data_dict['test']), 
        'ogt_rgr_train_mean': train_mean, 'ogt_rgr_train_std': train_std, 'ogt_rgr_test_mean': test_mean, 'ogt_rgr_test_mean': train_std}

    # save metrics
    with open('./data/ogt_protein_regressor/data_metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
