"""Trains and tests a classifier of OGT.

Parameters:
- `model`: 'protbert' or 'DeepTP', which model to start with
- `protocol`: 'head' or 'finetune'
  - head: will only train a classifier layer head to the base model
  - finetune: allows predictor head and base model to be backpropegated
  - bighead: like head but a deeper MLP used for classification
- `batch_size`: int, batch size for training and evaluation
- `epochs`: int, total epochs to train, best model is reloaded at the end
- `lr`: float, learning rate
- `lr_cheduler`: name og HF LR scheduler like 'linear'
- `n_save_per_epoch`: int, number of times to evaluate and save model per training epoch. 1 is once at the end of the epoch.
- `grad_checkpointing`: bool, whehert to train with grad checkpointing
- `grad_accum`: int, number of gradients to accumulate before backprop
- `dev_subsample_data`: int, whether downsample the training data to a desired size
"""
import argparse
import os
from yaml import safe_load as yaml_load
from yaml import dump as yaml_dump
import pandas as pd
import numpy as np
import re

import datasets
import transformers
import torch
import evaluate

import codecarbon
import dvclive

if 'SLURM_NTASKS' in os.environ:
    CPU_COUNT = int(os.environ['SLURM_NTASKS'])
else:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()

import l2tml_utils.model_utils as model_utils

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

    # get process rank
    # this is expected by pytorch to run distributed https://pytorch.org/docs/stable/elastic/run.html
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)['ogt_protein_classifier_train_evaluate']
    params['_data_batch_size'] = int(params['batch_size']/CPU_COUNT)
    logger.info(f"Loaded parameters: {params}")
    ds_batch_params = dict(batched=True, batch_size=params['_data_batch_size'], num_proc=CPU_COUNT)
    

    # prepare the main process
    if local_rank not in [-1, 0]:
        torch.distributed.barrier() # non main processes will stop here until the main process co
    else: # only main processes will run here
        # start dvc live. if done later it breaks logs
        live = dvclive.Live(dir='./data/ogt_protein_classifier/dvclive/', dvcyaml=False, report='md')
        # loggers
        logger.setLevel(getattr(logging, LOGLEVEL))
        fh = logging.FileHandler(LOGFILE, mode='w')
        formatter = logging.Formatter('%(filename)-12s %(asctime)s;%(funcName)-12s: %(levelname)-8s %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        utils_logger = logging.getLogger('l2tml_utils')
        utils_logger.setLevel(getattr(logging, LOGLEVEL))
        utils_logger.addHandler(fh)
    
        # create dirs
        if not os.path.exists('./data/ogt_protein_classifier/model'):
            os.mkdir('./data/ogt_protein_classifier/model')

        # start carbon tracker for data processing
        tracker = codecarbon.OfflineEmissionsTracker( 
            project_name="train_classifier",
            output_dir="./data/",
            country_iso_code="USA",
            region="washington"
        )
        tracker.start()

    # all processes will run here, but the main process will run first
    # so the others can just use the cache

    # get devices
    logger.info(f'Using {CPU_COUNT} cpus for multiprocessing')

    # get the data
    data_dict = datasets.load_from_disk('./data/ogt_protein_classifier/data')
    logger.info(f"Loaded data: {data_dict}")

    # drop spurious columns
    bad_columns = [c for c in data_dict['train'].column_names if c not in ['protein_seq', 'labels']]
    data_dict = data_dict.remove_columns(bad_columns)

    # DEV OPTION IGNORE FOR NORMAL OPERATION
    ########################################
    # cut training data to a single batch to check if we can overfit
    if params["dev_subsample_data"]:
        if params["dev_subsample_data"] < len(data_dict['train']):
            data_dict['train'] = data_dict['train'].select(range(params["dev_subsample_data"]))
        if params["dev_subsample_data"] < len(data_dict['test']):
            data_dict['test'] = data_dict['test'].select(range(params["dev_subsample_data"]))
        logger.info(f"Downsample train and test, now sizes {(len(data_dict['train']),len(data_dict['test']))}")
    ########################################

    # class weighting for imbalance
    train_positives = data_dict['train'].map(lambda e: {'sum': [sum(e['labels'])]}, **ds_batch_params, remove_columns=data_dict['train'].column_names, desc="Counting train positives")
    train_positives = sum(train_positives['sum'])
    positive_balance = train_positives/len(data_dict['train'])
    class_weight = [positive_balance/(1-positive_balance), 1.0]
    class_weight=torch.tensor(class_weight, dtype=torch.float).to(device)
    logger.info(f"Weights for class balancing: {class_weight}")
    
    # data format
    data_dict = data_dict.with_format("torch")
 
    # initialize the model
    if params['model'] == 'protbert':
        # load tokenizer and model
        # https://huggingface.co/Rostlab/prot_bert
        config = transformers.BertConfig.from_pretrained("Rostlab/prot_bert")
        
        # set hyperparam changes to config
        config.num_labels = 2
        # huggingface trainer sets the whole model to .train() each training step,
        # so we cannot just use .eval() on the model now to turn of bert dropout for a head only model
        # instead manualyl set bert dropout to 0 for a head model.
        if not params['protocol'] in ['head', 'bighead']:
            config.hidden_dropout_prob = params['dropout']
            config.attention_probs_dropout_prob = params['dropout']
        else:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        # load model
        # first get model class
        if params['protocol'] != 'bighead':
            model_class = transformers.BertForSequenceClassification
        else:
            model_class = model_utils.BertForSequenceClassificationBigHead
        # next check if we are starting from an internal checkpoint of the
        # HF hub one
        # get the checkpoints on disk that are not empty
        checkpoints  = [f for f in os.listdir('./data/ogt_protein_classifier/model/') if f.startswith('checkpoint')]
        checkpoints = [f for f in checkpoints if len(os.listdir('./data/ogt_protein_classifier/model/'+f)) > 0]
        if len(checkpoints) == 0:
            logger.info("Using original Protein Bert weights")
            pretrained_weights_location = "Rostlab/prot_bert"
        else:
            checkpoints_nums = [int(c.split('-')[-1]) for c in checkpoints]
            pretrained_weights_location = checkpoints[np.argmax(checkpoints_nums)]
            pretrained_weights_location = './data/ogt_protein_classifier/model/'+pretrained_weights_location
            logger.info(f"Using weights loaded from local checkpoint: {pretrained_weights_location}")
        model = model_class.from_pretrained(
            pretrained_weights_location, config=config
        )

        # and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        logger.info(f"Loaded ProtBERT model and tokenizer. Model config: {model.config}")

        # tokenize the data
        def prepare_aa_seq(examples):
            seqs = [' '.join(e[1:]) for e in examples['protein_seq']]
            seqs = [re.sub(r"[UZOB]", "X", e) for e in seqs]
            examples['protein_seq'] = seqs
            return examples
        data_dict = data_dict.map(prepare_aa_seq, **ds_batch_params, desc="Reformatting AA sequences for BERT")
        logger.info('Prepared sequences appropriate for Prot BERT: No M start, spaces between AA, sub UZOB with X')

        def tokenizer_fn(examples):
            return tokenizer(examples["protein_seq"], max_length=512, padding="max_length", truncation=True)
        data_dict = data_dict.map(tokenizer_fn, **ds_batch_params, desc="Tokenizing data")
        data_dict = data_dict.remove_columns('protein_seq')
        logger.info(f'Tokenized dataset. {data_dict}')
        logger.info(f"Dataset snippet: {data_dict['train'][:5]}")

        # fix the model if necessary
        if params['protocol'] in ['head', 'bighead']:
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
    else:
        raise NotImplementedError(f"Model type {params['model']} not available") 

    # allow main process to barrier out so the other processes can catch up
    if local_rank == 0:
        torch.distributed.barrier() 

    # compute the saving and evaluation timeframe
    n_gpus = torch.cuda.device_count()
    if params['grad_accum']:
        grad_accum = params['grad_accum']
    else:
        grad_accum = 1
    total_data_per_step = params['batch_size']*grad_accum*n_gpus
    steps_per_epoch = int(len(data_dict['train'])/total_data_per_step)

    if params['n_save_per_epoch'] == 0:
        n_steps_per_save = None
        save_strategy = 'no'
    else:
        save_strategy = 'steps'
        n_steps_per_save = int(steps_per_epoch/params['n_save_per_epoch'])
    logger.info(f"Saving/evaluating every {n_steps_per_save} steps of size {total_data_per_step}")
    
    # ready the train
    training_args = transformers.TrainingArguments(
        do_train=True,
        do_eval=True,
        optim='adamw_hf',
        optim_args=None,
        learning_rate=float(params['lr']),
        lr_scheduler_type=params['lr_scheduler'],
        num_train_epochs=params['epochs'],
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        gradient_accumulation_steps=params['grad_accum'],
        eval_accumulation_steps=params['grad_accum'],
        gradient_checkpointing=params['grad_checkpointing'],
        fp16=params['fp16'],
        log_level='info',
        logging_strategy='steps',
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=n_steps_per_save,
        evaluation_strategy=save_strategy,
        eval_steps=n_steps_per_save,
        output_dir='./data/ogt_protein_classifier/model',
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    def compute_metrics(eval_pred):
        f1=evaluate.load('f1')
        acc=evaluate.load('accuracy')
        matt=evaluate.load('matthews_correlation')
        cfm = evaluate.load("BucketHeadP65/confusion_matrix")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1_val = f1.compute(predictions=predictions, references=labels)['f1']
        acc_val = acc.compute(predictions=predictions, references=labels)['accuracy']
        matt_val = matt.compute(predictions=predictions, references=labels)['matthews_correlation']
        cfm_val = list(cfm.compute(predictions=predictions, references=labels)['confusion_matrix'].flatten())
        return {'f1': f1_val, 'accuracy':acc_val, 'matthew': matt_val, 'cfm': cfm_val}
    
    # set up a dvccallback
    if local_rank in [-1, 0]:
        callbacks = [model_utils.DVCLiveCallback(live)]
    else:
        callbacks = None

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
        callbacks=callbacks
    )
    logger.info(f"Training parameters ready: {training_args}, beginning.")
    
    # remove the codecarbon callback, doing this manually
    trainer.remove_callback(transformers.integrations.CodeCarbonCallback)

    # run it!
    training_results = trainer.train()
    logger.info(f"Training results: {training_results}")
    training_log = pd.DataFrame(trainer.state.log_history[:-1]).to_dict(orient='list')

    # test it
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")
    
    # only main process records results
    if local_rank in [-1, 0]:
        if params['push']:
            model.push_to_hub('learn2therm')
            tokenizer.push_to_hub('learn2therm')
        # save model
        model.save_pretrained('./data/ogt_protein_classifier/model')
        tokenizer.save_pretrained('./data/ogt_protein_classifier/model')

        # add end of training metrics
        metrics=dict(eval_result)
        metrics.update(dict(training_results.metrics))
        metrics.update(training_log)
        emissions=float(tracker.stop())
        metrics['emissions'] = emissions
        metrics = {'ogt_cfr_model_'+k: v for k, v in metrics.items()}

        # save metrics
        with open('./data/ogt_protein_classifier/model_metrics.yaml', "w") as stream:
            yaml_dump(metrics, stream)
