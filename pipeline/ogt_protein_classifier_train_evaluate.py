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
import re

from datasets import Dataset
import transformers
import torch
import evaluate
import sklearn.utils

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
    formatter = logging.Formatter('%(filename)-12s %(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # get device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info('Using device {device}')

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
        USING SAMPLE 10000""",
        config_name='test',
        cache_dir='./tmp/hf_cache',
        con=conn)
    logger.info(f"Loaded data from database, {len(ds)} total points")

    # remove no OGT label
    ds = ds.filter(lambda e: e['ogt'] != None)
    logger.info(f"Removed examples without OGT, {len(ds)} datapoints remaining")

    # determine label window
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

    # data balance weighting
    if params['balance']:
        classes = [0,1]
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            'balanced',
            classes=classes,
            y=ds['label']
        )
        class_weight = dict(zip(classes, class_weight))
        logger.info(f"Weights for class balancing: {class_weight}")
    else:
        class_weight = None
        logger.info(f"Not considering class balance during training.")
    
    # split the data
    splitter = data_utils.DataSplitter(ds)
    data_dict = splitter.split(splittype=params['split_type'])
    logger.info(f"Split data into train and test")
    logger.info(f"Train balance: {sum(data_dict['train']['label'])/len(data_dict['train'])}")
    logger.info(f"Test balance: {sum(data_dict['test']['label'])/len(data_dict['test'])}")
    
    # remove unnecessary columns
    data_dict = data_dict.map(lambda e: e, remove_columns=['protein_int_index', 'ogt', 'taxa_index', 'taxonomy'])
    logger.info(f'Final datasets: {data_dict}')
    data_dict.save_to_disk('./data/ogt_protein_classifier/data/')
    logger.info("Saved data to disk.")
    
    # initialize the model
    if params['model'] == 'protbert':
        # load tokenizer and model
        # https://huggingface.co/Rostlab/prot_bert
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "Rostlab/prot_bert"
        )
        model.to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        logger.info("Loaded ProtBERT model and tokenizer")

        # tokenize the data
        def prepare_aa_seq(example):
            example['protein_seq'] = ' '.join(example['protein_seq'][1:])
            example['protein_seq'] = re.sub(r"[UZOB]", "X", example['protein_seq'])
            return example
        data_dict = data_dict.map(prepare_aa_seq)
        logger.info('Prepared sequences appropriate for Prot BERT: No M start, spaces between AA, sub UZOB with X')

        def tokenizer_fn(examples):
            return tokenizer(examples["protein_seq"], max_length=512, padding="max_length", truncation=True)
        data_dict = data_dict.map(tokenizer_fn, batched=True)
        logger.info('Tokenized dataset.')
        print(data_dict)

        # fix the model if necessary
        if params['protocol'] == 'head':
            for param in model.bert.parameters():
                param.requires_grad=False
            logger.info(f'Fixing all but classifer head for finetuning.')
        else:
            logger.info('Leaving whole model trainable.')
        
        # check the parameters and log
        total_params = 0
        trainable_params = 0
        for param in model.parameters():
            num = np.prod(param.size())
            total_params += num
            if param.requires_grad:
                trainable_params += num
        logger.info(f'{total_params} total params, {trainable_params} trainable.')
        
        # ready the train
        training_args = transformers.TrainingArguments(
            label_names=['label'],
            optim='adamw_hf',
            optim_args=None,
            learning_rate=5e-5,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            log_level='info',
            logging_strategy='epoch',
            save_strategy='epoch',
            output_dir='./data/ogt_protein_classifier'
        )
        f1 = evaluate.load("f1")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {'f1': f1.compute(predictions=predictions, references=labels)}

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=data_dict['train'],
            eval_dataset=data_dict['test'],
            compute_metrics=compute_metrics,
        )
        logger.info(f"Training parameters ready: {training_args}, beginning.")
        # run it!
        trainer.train()
        
        # test it
        trainer.evaluate()

    else:
        raise NotImplementedError(f"Model type {params['model']} not available")
        
    