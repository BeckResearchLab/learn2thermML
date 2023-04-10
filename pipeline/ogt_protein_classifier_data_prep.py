"""Trains and tests a classifier of OGT.

Parameters:
- `split_type`: 'random', 'taxa_id', int
  - random: randomly splits proteins into train and dev
  - taxa_id: keeps proteins from a particular organism together
  - int: clusters taxa by taxonomy of a particular level, 1 is kingdom, 2 is phylum, etc. Note that this may have errors, as it is based on taxonomy specified in NCBI which sometimes has ambiquity
- `ogt_window`: (float, float), low and high temparature of window between OGT classes
- `max/min_protein_len`: int, maximum and minumum protein length to consider
- `min_balance`: float or None, minimum balance of classes. class weighted training is always used, but this can be used to speicify downsampling of the 
    majority class in order to meet a minimum imbalance
- `max_upsampling`: float, maximum fraction of original minority class to insert into the dataset
    ignored if `min_balance` is None
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
    if not os.path.exists('./data/ogt_protein_classifier/data'):
        os.mkdir('./data/ogt_protein_classifier/data')

    # device settings
    logger.info(f'Using {CPU_COUNT} cpus for multiprocessing')

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)['ogt_protein_classifier_data_prep']
    logger.info(f"Loaded parameters: {params}")
    dedup_dict = {}
    for d in params['deduplication']:
        dedup_dict.update(d)
    params['deduplication'] = dedup_dict
    
    # set some kwargs for datasets batching
    ds_batch_params = dict(batched=True, batch_size=params['data_batch_size'], num_proc=CPU_COUNT)
    
    # start carbon tracker for data processing
    data_tracker = codecarbon.OfflineEmissionsTracker( 
        project_name="data_prep_classifier",
        output_dir="./data/",
        country_iso_code="USA",
        region="washington"
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

    # determine label window
    low = params['ogt_window'][0]
    high = params['ogt_window'][1]
    def get_label(examples):
        ogts = np.array(examples['ogt']).reshape(-1,1)
        labels = np.ones(ogts.shape) * -1
        labels[ogts <= low] = 0
        labels[ogts >= high] = 1
        examples['labels'] = list(labels.reshape(-1).astype(int))
        return examples
    ds = ds.map(get_label, **ds_batch_params, desc="Applying labels to raw OGT")
    logger.info('Labeled examples...')
    ds = ds.filter(lambda e: list(~np.isclose(np.array(e['labels']), -1)), **ds_batch_params, desc="Removing examples in OGT window")
    logger.info(f'Removed examples within window, {len(ds)} datapoints remaining')
    total_positives = ds.map(lambda e: {'sum': [sum(e['labels'])]}, **ds_batch_params, remove_columns=ds.column_names, desc="Counting total number of positives")
    logger.info(f'Initial dataset balance: {sum(total_positives["sum"])/len(ds)}')

    # class balancing
    if params['min_balance'] is not None:
        # split the train into positive and negative
        logger.info(f"Beginning balancing")
        positives = ds.filter(lambda e: list(np.isclose(e['labels'], 1)), **ds_batch_params, desc="Splitting out positives for balancing")
        positives = positives.shuffle()
        negatives = ds.filter(lambda e: list(np.isclose(e['labels'], 0)), **ds_batch_params, desc="Splitting our negatives for balancing")
        negatives = negatives.shuffle()
        # get the suggested class data sizes
        logger.info(f"Conducting balancing on data with {len(positives)} positives and {len(negatives)} negative.")
        n_negative, n_positive = data_utils.get_balance_data_sizes(
            len(negatives),
            len(positives),
            desired_balance=params['min_balance'],
            max_upsampling=params['max_upsampling'])

        # actualy sample it
        data_dict = {'negative': negatives, 'positive': positives}
        desired_balance_dict = {f'negative': n_negative, f'positive': n_positive}
        for class_, n_class in desired_balance_dict.items():
            if n_class < len(data_dict[class_]):
                # we can just select the first n since its already shuffled for downsampling
                data_dict[class_] = data_dict[class_].select(range(n_class))
            elif n_class > len(data_dict[class_]):
                # sample with replacement to upsample
                indexes = np.random.randint(0, len(data_dict[class_]), size=n_class)
                data_dict[class_] = data_dict[class_].select(indexes)
            else:
                pass
        positives = data_dict['positive']; negatives = data_dict['negative']
        logger.info(f"Final negative, positive classsizes: {len(positives)}, {len(negatives)}")
        # stick the data back together
        ds = datasets.concatenate_datasets([positives, negatives]).shuffle()

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
    
    # split the data
    splitter = data_utils.DataSplitter(ds)
    data_dict = splitter.split(splittype=params['split_type'], frac=params['train_test_frac'])
    data_dict = data_dict.shuffle()
    logger.info(f"Split data into train and test")
    train_positives = data_dict['train'].map(lambda e: {'sum': [sum(e['labels'])]}, **ds_batch_params, remove_columns=data_dict['train'].column_names, desc="Counting train positives")
    test_positives = data_dict['test'].map(lambda e: {'sum': [sum(e['labels'])]}, **ds_batch_params, remove_columns=data_dict['test'].column_names, desc="Counting test positives")
    train_balance = sum(train_positives['sum'])/len(data_dict['train'])
    test_balance = sum(test_positives['sum'])/len(data_dict['test'])
    logger.info(f"Train balance: {train_balance}")
    logger.info(f"Test balance: {test_balance}")
    

    # remove unnecessary columns
    if not params['dev_keep_columns']:
        data_dict = data_dict.remove_columns(['protein_int_index', 'ogt', 'taxa_index', 'taxonomy'])
    logger.info(f'Final datasets: {data_dict}')
    data_dict.cleanup_cache_files()
    data_dict.save_to_disk('./data/ogt_protein_classifier/data/')
    logger.info("Saved data to disk.")
    
    # get co2
    co2 = data_tracker.stop()

    metrics = {'ogt_cfr_data_co2': co2, 'ogt_cfr_data_n_train': len(data_dict['train']), 'ogt_cfr_data_n_test': len(data_dict['test']), 'ogt_cfr_bal_train': train_balance, 'ogt_cfr_bal_test': test_balance}

    # save metrics
    with open('./data/ogt_protein_classifier/data_metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
