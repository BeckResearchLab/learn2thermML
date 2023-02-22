"""Trains and tests a classifier of OGT.

Parameters:
- `split_type`: 'random', 'taxa_id', int
  - random: randomly splits proteins into train and dev
  - taxa_id: keeps proteins from a particular organism together
  - int: clusters taxa by taxonomy of a particular level, 1 is kingdom, 2 is phylum, etc. Note that this may have errors, as it is based on taxonomy specified in NCBI which sometimes has ambiquity
- `ogt_window`: (float, float), low and high temparature of window between OGT classes
- `min_balance`: float or None, minimum balance of classes. class weighted training is always used, but this can be used to speicify downsampling of the 
    majority class in order to meet a minimum imbalance
- `max_upsampling`: float, maximum fraction of original minority class to insert into the dataset
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
import codecarbon

import l2tml_utils.data_utils as data_utils

import logging
logger = logging.getLogger(__name__)

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
    logger.setLevel(getattr(logging, LOGLEVEL))
    fh = logging.FileHandler(LOGFILE, mode='w')
    formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(getattr(logging, LOGLEVEL))
    transformers_logger.addHandler(fh)
    utils_logger = logging.getLogger('l2tml_utils')
    utils_logger.setLevel(getattr(logging, LOGLEVEL))
    utils_logger.addHandler(fh)
    # datasets.utils.logging.set_verbosity('WARNING')
    # transformers.utils.logging.set_verbosity(LOGLEVEL)
    
    # create dirs
    if not os.path.exists('./data/ogt_protein_classifier/model'):
        os.mkdir('./data/ogt_protein_classifier/model')
    if not os.path.exists('./data/ogt_protein_classifier/data'):
        os.mkdir('./data/ogt_protein_classifier/data')

    # get device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using device {device}')

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)['ogt_protein_classifier_train_evaluate']
    logger.info(f"Loaded parameters: {params}")
    
    # start carbon tracker for data processing
    data_tracker = codecarbon.EmissionsTracker( 
        project_name="data_process",
        output_dir="./data/ogt_protein_classifier/model",
    )
    data_tracker.start()

    # get the data
    conn = ddb.connect("./data/database")
    # create indexes first
    conn.execute("CREATE UNIQUE INDEX taxa_index ON taxa (taxa_index)")
    conn.commit()
    conn.execute("CREATE INDEX taxa_index_foreign ON proteins (taxa_index)")
    conn.commit()
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
    conn.close()

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
        # get the suggested class data sizes
        logger.info(f"Conducting balancing for training data with {len(data_dict['positive'])} positive and {len(data_dict['negative'])} negative.")
        n_negative, n_positive = data_utils.get_balance_data_sizes(
            len(data_dict['negative']),
            len(data_dict['positive']),
            desired_balance=params['min_balance'],
            max_upsampling=params['max_upsampling'])

        # actualy sample it
        desired_balance_dict = {'negative': n_negative, 'positive': n_positive}
        for class_, n_class in desired_balance_dict.items():
            if n_class < len(data_dict[class_]):
                # we can just select the first n since its already shuffled for downsampling
                data_dict[class_] = data_dict[class_].select(range(n_negative))
            elif n_class > len(data_dict[class_]):
                # sample with replacement to upsample
                indexes = np.random.randint(0, len(data_dict[class_]), size=n_class)
                data_dict[class_] = data_dict[class_].select(indexes)
            else:
                pass
        logger.info(f"Final negative, positive class training sizes: {len(data_dict['negative'])}, {len(data_dict['positive'])}")
        # stick the data back together
        data_dict['train'] = datasets.concatenate_datasets([data_dict['positive'], data_dict['negative']]).shuffle()
        # drop the datasets from processing
        _ = data_dict.pop('positive')
        _.cleanup_cache_files()
        del(_)
        _ = data_dict.pop('negative')
        _.cleanup_cache_files()
        del(_)

        
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
    data_dict.cleanup_cache_files()
    data_dict.save_to_disk('./data/ogt_protein_classifier/data/')
    logger.info("Saved data to disk.")
    
    # sending data
    data_dict = data_dict.with_format("torch")
 
    # initialize the model
    if params['model'] == 'protbert':
        # load tokenizer and model
        # https://huggingface.co/Rostlab/prot_bert
        config = transformers.BertConfig.from_pretrained("Rostlab/prot_bert")
        
        # set hyperparam changes to config
        config.num_labels=2
        config.classifier_dropout = params['dropout']
        # huggingface trainer sets the whole model to .train() each training step,
        # so we cannot just use .eval() on the model now to turn of bert dropout for a head only model
        # instead manualyl set bert dropout to 0 for a head model.
        if not params['protocol'] == 'head':
            config.hidden_dropout_prob = params['dropout']
            config.attention_probs_dropout_prob = params['dropout']
        else:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        # load model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "Rostlab/prot_bert", config=config
        )

        # and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        logger.info(f"Loaded ProtBERT model and tokenizer. Model config: {model.config}")

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
        
        # get data processing emissions
        data_emissions = data_tracker.stop()

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
        
        # compute the saving and evaluation timeframe
        n_steps_per_epoch = int(len(data_dict['train']) / params['batch_size'])
        n_steps_per_save = int(n_steps_per_epoch/params['n_save_per_epoch'])
        logger.info(f"Saving/evaluating every {n_steps_per_save} batches of size {params['batch_size']}")

        # ready the train
        training_args = transformers.TrainingArguments(
            do_train=True,
            do_eval=True,
            optim='adamw_hf',
            optim_args=None,
            learning_rate=5e-5,
            num_train_epochs=params['epochs'],
            per_device_train_batch_size=params['batch_size'],
            per_device_eval_batch_size=params['batch_size'],
            log_level='info',
            logging_strategy='steps',
            logging_steps=1,
            save_strategy='steps',
            save_steps=n_steps_per_save,
            evaluation_strategy='steps',
            eval_steps=n_steps_per_save,
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
        
        # send model to device and go
        model.to(device)
        logger.info(f"Model ready for training: {model}")
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
        emissions=callback.tracker.final_emissions
        metrics['model_emissions'] = emissions
        metrics['data_emissions'] = data_emissions

    else:
        raise NotImplementedError(f"Model type {params['model']} not available")

    # save metrics
    with open('./data/ogt_protein_classifier/metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
