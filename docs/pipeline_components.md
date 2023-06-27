# DVC Pipeline steps for producing a neural translator of low tempt to high temp proteins

## OGT Classifier Of Proteins
Trains and evaluates a classifier of OGT based on protein sequence.

### Script: `ogt_protein_classifier_data_prep.py`

Prepares data for training and evaluating LPLM classifier

### Params
- `split_type`: 'random', 'taxid', int - How to split train and test data            
- `ogt_window`: tuple, window of OGT in celsius to ignore in data
- `max_protein_len`: int
- `min_protein_len`: int
- `train_test_frac`: portion of data to make test
- `deduplication`: 
    - `do`: bool, whether to run deduplication
    - `jaccard`: minimum jaccard to consider two sequences similar
    - `kgram`: int, word size of kgram representation of sequences
    - `num_perm:` MinHash number of permutations
- `min_balance`: float, minumum fraction of minority class
- `max_upsampling`: float, maximum fraction of minorioty class oto upsample

### Script: `ogt_protein_classifier_train_evaluate.py`
Train and evaluate the classifier.

### Params
- `model`: 'protbert', pretrained model to use
- `protocol`: 'finetune', 'head', 'bighead', whether to finetune whole LPLM, or only train a head classifier
- `dropout`: float, dropout rate
- `batch_size`: int, batch size
- `epochs`: int, number of epochs to train
- `lr`: float, learning rate
- `lr_scheduler`: 'cosine', 'linear', 'constant', learning rate scheduler
- `n_save_per_epoch`: int, number checkpoints per epoch
- `grad_checkpointing`: bool, whether to use gradient checkpointing
- `grad_accum`: int, number of batches to accumulate gradients over
- `fp16`: bool, whether to use mixed precision training
- `push`: bool, whether to push model to huggingface hub