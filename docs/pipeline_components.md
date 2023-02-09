# DVC Pipeline steps for producing a neural translator of low tempt to high temp proteins

## OGT Classifier Of Proteins
Trains and evaluates a classifier of OGT based on protein sequence.

### Script: `ogt_protein_classifier_train_evaluate.py`

### Params
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