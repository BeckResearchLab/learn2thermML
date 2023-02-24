import transformers
from torch import nn


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
