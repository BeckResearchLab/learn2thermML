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
- `dev_overtrain_one_batch`: bool, whether to make the whole dataset a single batch
"""
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

if 'SLURM_CPUS_ON_NODE' in os.environ:
    CPU_COUNT = int(os.environ['SLURM_CPUS_ON_NODE'])
else:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()

import l2tml_utils.data_utils as model_utils

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
    
    # create dirs
    if not os.path.exists('./data/ogt_protein_classifier/model'):
        os.mkdir('./data/ogt_protein_classifier/model')

    # get device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using device {device} for ML')
    logger.info(f'Using {CPU_COUNT} cpus for multiprocessing')

    # load parameters
    with open("./params.yaml", "r") as stream:
        params = yaml_load(stream)['ogt_protein_classifier_train_evaluate']
    params['_data_batch_size'] = int(params['batch_size']/CPU_COUNT)
    logger.info(f"Loaded parameters: {params}")
    ds_batch_params = dict(batched=True, batch_size=params['_data_batch_size'], num_proc=CPU_COUNT)
    
    # start carbon tracker for data processing
    tracker = codecarbon.EmissionsTracker( 
        project_name="model_train",
        output_dir="./data/ogt_protein_classifier/model",
    )
    tracker.start()

    # get the data
    data_dict = datasets.load_from_disk('./data/ogt_protein_classifier/data')
    logger.info(f"Loaded data: {data_dict}")

    # DEV OPTION IGNORE FOR NORMAL OPERATION
    ########################################
    # cut training data to a single batch to check if we can overfit
    if params["dev_overtrain_one_batch"]:
        data_dict['train'] = data_dict['train'].select(range(params['batch_size']))
        data_dict['test'] = data_dict['test'].select(range(params['batch_size']))
        logger.info(f"Overfitting test on a single example...")
    ########################################

    # class weighting for imbalance
    train_positives = data_dict['train'].map(lambda e: {'sum': [sum(e['labels'])]}, **ds_batch_params, remove_columns=data_dict['train'].column_names, desc="Counting train positives")
    train_positives = sum(train_positives['sum'])
    positive_balance = train_positives/len(data_dict['train'])
    class_weight = [positive_balance/(1-positive_balance), 1.0]
    class_weight=torch.tensor(class_weight, dtype=torch.float).to(device)
    logger.info(f"Weights for class balancing: {class_weight}")
    
    # sending data
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
        if params['protocol'] != 'bighead':
            model_class = transformers.BertForSequenceClassification
        else:
            import l2tml_utils.model_utils
            model_class = l2tml_utils.model_utils.BertForSequenceClassificationBigHead
        model = model_class.from_pretrained(
            "Rostlab/prot_bert", config=config
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
        
        # compute the saving and evaluation timeframe
        n_steps_per_epoch = int(len(data_dict['train']) / params['batch_size'])
        if params['grad_accum']:
            n_steps_per_epoch = int(n_steps_per_epoch/params['grad_accum'])
        if params['n_save_per_epoch'] == 0:
            n_steps_per_save = None
            save_strategy = 'no'
        else:
            save_strategy = 'steps'
            n_steps_per_save = int(n_steps_per_epoch/params['n_save_per_epoch'])
        logger.info(f"Saving/evaluating every {n_steps_per_save} batches of size {params['batch_size']}")

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
        
        # remove the codecarbon callback, doing this manually
        trainer.remove_callback(transformers.integrations.CodeCarbonCallback)

        # run it!
        training_results = trainer.train()
        logger.info(f"Training results: {training_results}")
        training_log = pd.DataFrame(trainer.state.log_history[:-1]).to_dict(orient='list')

        # test it
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")
        
        # save model
        model.save_pretrained('./data/ogt_protein_classifier/model')

        # add other metrics
        metrics=dict(eval_result)
        metrics.update(dict(training_results.metrics))
        metrics.update(training_log)
        emissions=float(tracker.stop())
        metrics['emissions'] = emissions
        metrics = {'ogt_cfr_model_'+k: v for k, v in metrics.items()}

    else:
        raise NotImplementedError(f"Model type {params['model']} not available")

    # save metrics
    with open('./data/ogt_protein_classifier/model_metrics.yaml', "w") as stream:
        yaml_dump(metrics, stream)
