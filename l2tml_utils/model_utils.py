"""Children classes of huggingface functionality to be used for training"""

import transformers
from torch import nn

import dvclive
from dvclive.utils import standardize_metric_name

from typing import Optional



class BertForSequenceClassificationBigHead(transformers.BertForSequenceClassification):
    """Huggingface's Bert classifier model execpt the head model is an MLP instead of a 2 neuron linear classifier."""
    def __init__(self, config):
        # super up higher
        super(transformers.BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = transformers.BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # add a big head instead of a single linear layer. This is what differs from the
        # has 2 layers
        # normal bert classifier
        classifier = []
        classifier.append(nn.Linear(config.hidden_size, config.hidden_size))
        classifier.append(nn.Tanh())
        classifier.append(nn.Dropout(config.classifier_dropout))
        classifier.append(nn.Linear(config.hidden_size, int(config.hidden_size/2)))
        classifier.append(nn.Tanh())
        classifier.append(nn.Dropout(config.classifier_dropout))
        classifier.append(nn.Linear(int(config.hidden_size/2), config.num_labels, bias=False))
        self.classifier = nn.Sequential(*classifier)

        # Initialize weights and apply final processing
        self.post_init()


class DVCLiveCallback(transformers.TrainerCallback):
    """HF callback for use with DVCLive.
    
    DVC's implimentation own implimentation has the following non-ideal features:
    - dvc live step is each logging step, instead of each eval step
    - it manually saves model at each epoch. 

    Here, on evaluate the metrics are recorded, and triggers the trainer control to
    save at this gradient ste. Then, on save, the dvclive step is taken.
    """
    
    def __init__(self, live: Optional[dvclive.Live] = None, **kwargs):
        super().__init__()
        self.live = live if live is not None else dvclive.Live(**kwargs)

    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        """DVC should checkpoint"""
        self.live.next_step()

    def on_log(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        # saves training status as metrics to dvc, note that 
        # this only occurs when an evaluation step is necessary
        # logs are not available in the on_evaluate event
        logs = kwargs["logs"]
        if control.should_evaluate:
            for key, value in logs.items():
                try:
                    self.live.log_metric(standardize_metric_name(key, __name__), value)
                except:
                    pass # some things floating in the logs may not be recordable by dvc

    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        metrics = kwargs.get('metrics')
        for key, value in metrics.items():
            try:
                self.live.log_metric(standardize_metric_name(key, __name__), value)
            except:
                pass # some things floating in the logs may not be recordable by dvc
        # if not already going to save, make sure it does
        # on_save above will be called and next step starts
        control.should_save=True

    def on_train_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs
    ):
        self.live.end()
