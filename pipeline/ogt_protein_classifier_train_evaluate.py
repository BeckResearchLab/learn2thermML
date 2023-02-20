"""Trains and tests a classifier of OGT.

Parameters:
- `split_type`: 'random', 'taxa_id', int
  - random: randomly splits proteins into train and dev
  - taxa_id: keeps proteins from a particular organism together
  - int: clusters taxa by taxonomy of a particular level, 1 is kingdom, 2 is phylum, etc. Note that this may have errors, as it is based on taxonomy specified in NCBI which sometimes has ambiquity
- `ogt_window`: (float, float), low and high temparature of window between OGT classes
- `min_balance`: float or None, minimum balance of classes. class weighted training is always used, but this can be used to speicify downsampling of the 
    majority class in order to meet a minimum imbalance
- `upsample_frac`: float, fraction of data balancing that will be conducted by upsampling the minority class, the rest is downsampling
    ignored if `min_balance` is None
- `model`: 'protbert' or 'DeepTP', which model to start with
- `protocol`: 'head' or 'finetune'
  - head: will only train a MLP head to the base model
  - finetune: allows predictor head and base model to be backpropegated
- `batch_size`: int, batch size for training and evaluation
- `epochs`: int, total epochs to train, best model is reloaded at the end
- `n_save_per_epoch`: int, number of times to evaluate and save model per training epoch. 1 is once at the end of the epoch.

"""
import os
from yaml import safe_load as yaml_load
from yaml import dump as yaml_dump
import pandas as pd
import numpy as np
import duckdb as ddb
import re

import datasets
import transformers
import torch
import evaluate
import sklearn.utils

import data_utils

import logging

if 'LOGLEVEL' in os.environ:
    LOGLEVEL = os.environ['LOGLEVEL']
else:
    LOGLEVEL = 'INFO'
LOGNAME = __file__
LOGFILE = f'./logs/{os.path.basename(__file__)}.log'

class ImbalanceTrainer(transformers.Trainer):
    """Trainer using cross entropy loss with imbalanced classes."""
    def __init__(self, class_weights, *args, **kwargs):
        self._class_weights = class_weights
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self._class_weights is None:
            return super().compute_loss(model, inputs, return_outputs)
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self._class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    # start logger
    logger = logging.getLogger(LOGNAME)
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    datasets.utils.logging.set_verbosity(LOGLEVEL)
    transformers.utils.logging.set_verbosity(LOGLEVEL)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # get device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using device {device}')

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)['ogt_protein_classifier_train_evaluate']
    logger.info(f"Loaded parameters: {params}")

    # get the data
    conn = ddb.connect("./data/database")
    ds = datasets.Dataset.from_sql(
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
        USING SAMPLE 5000""",
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
            example['labels']=0
        elif example['ogt']>=high:
            example['labels']=1
        else:
            example['labels']=None
        return example
    ds = ds.map(get_label)
    logger.info('Labeled examples...')
    ds = ds.filter(lambda e: e['labels'] != None)
    logger.info(f'Removed examples within window, {len(ds)} datapoints remaining')
    logger.info(f'Initial dataset balance: {sum(ds["labels"])/len(ds)}')
    
    # split the data
    splitter = data_utils.DataSplitter(ds)
    data_dict = splitter.split(splittype=params['split_type'])
    data_dict = data_dict.shuffle()
    logger.info(f"Split data into train and test")
    logger.info(f"Train balance: {sum(data_dict['train']['labels'])/len(data_dict['train'])}")
    logger.info(f"Test balance: {sum(data_dict['test']['labels'])/len(data_dict['test'])}")
    
    # class balancing
    if params['min_balance'] is not None:
        # split the train into positive and negative
        data_dict['positive'] = data_dict['train'].filter(lambda e: e['labels']==1)
        data_dict['positive'] = data_dict['positive'].shuffle()
        data_dict['negative'] = data_dict['train'].filter(lambda e: e['labels']==0)
        data_dict['negative'] = data_dict['negative'].shuffle()
        # which class is the majority?
        majority_class = 'positive' if len(data_dict['positive']) > len(data_dict['negative']) else 'negative' 
        minority_class = 'positive' if majority_class == 'negative' else 'negative'
        # what is the difference in data size?
        imbalance_size = len(data_dict[majority_class]) - len(data_dict[minority_class])
        n_upsample = int(imbalance_size*params['upsample_frac'])
        n_downsample = imbalance_size - n_upsample
        logger.info(f"Conducting balancing for training data with {len(data_dict['positive'])} postiive and {data_dict['negative']} negative. Majority: {majority_class}")
        logger.info(f"Data size difference: {imbalance_size}. Removing {} from majority and upsampling {} from minority")
        
    # class weighting for imbalance
    classes = [0,1]
    class_weight = sklearn.utils.class_weight.compute_class_weight(
        'balanced',
        classes=classes,
        y=data_dict['train']['labels']
    )
    class_weight=torch.tensor(class_weight, dtype=torch.float).to(device)
    logger.info(f"Weights for class balancing: {class_weight}")

    # remove unnecessary columns
    data_dict = data_dict.map(lambda e: e, remove_columns=['protein_int_index', 'ogt', 'taxa_index', 'taxonomy'])
    logger.info(f'Final datasets: {data_dict}')
    data_dict.save_to_disk('./data/ogt_protein_classifier/data/')
    logger.info("Saved data to disk.")
    
    # sending data
    data_dict = data_dict.with_format("torch")
 
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
        data_dict = data_dict.map(lambda e: e, remove_columns=['protein_seq'])
        logger.info(f'Tokenized dataset. {data_dict}')

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
            do_train=True,
            do_eval=True,
            optim='adamw_hf',
            optim_args=None,
            learning_rate=5e-5,
            num_train_epochs=3,
            per_device_train_batch_size=300,
            per_device_eval_batch_size=32,
            log_level='info',
            logging_strategy='steps',
            logging_steps=1,
            save_strategy='steps',
            evaluation_strategy='steps',
            eval_steps=5,
            output_dir='./data/ogt_protein_classifier/model',
            load_best_model_at_end=True
        )
        def compute_metrics(eval_pred):
            f1=evaluate.load('f1')
            acc=evaluate.load('accuracy')
            matt=evaluate.load('matthews_correlation')

            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            f1_val = f1.compute(predictions=predictions, references=labels)['f1']
            acc_val = acc.compute(predictions=predictions, references=labels)['accuracy']
            matt_val = matt.compute(predictions=predictions, references=labels)['matthews_correlation']

            return {'f1': f1_val, 'accuracy':acc_val, 'matthew': matt_val}

        trainer = ImbalanceTrainer(
            class_weights=class_weight,
            model=model,
            args=training_args,
            train_dataset=data_dict['train'],
            eval_dataset=data_dict['test'],
            compute_metrics=compute_metrics,
        )
        logger.info(f"Training parameters ready: {training_args}, beginning.")
        
        # run it!
        training_results = trainer.train()
        logger.info(f"Training results: {training_results}")

        # test it
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")

        # add other metrics
        metrics=dict(eval_result)
        callback = trainer.pop_callback(transformers.integrations.CodeCarbonCallback)
        emissions=callback.final_emissions
        metrics['emissions'] = emissions

        with open('./data/ogt_protein_classifier/metrics.yaml', "w") as stream:
                yaml_dump(metrics, stream)

    else:
        raise NotImplementedError(f"Model type {params['model']} not available")
        
    
    # save metrics
    with open('./data/ogt_protein_classifier/metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
