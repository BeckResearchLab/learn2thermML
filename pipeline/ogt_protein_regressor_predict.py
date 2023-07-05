import torch
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_from_disk
from sklearn.metrics import mean_squared_error, r2_score
import re

def prepare_aa_seq(examples):
    seqs = [' '.join(e[1:]) for e in examples['protein_seq']]
    seqs = [re.sub(r"[UZOB]", "X", e) for e in seqs]
    examples['protein_seq'] = seqs
    return examples

# get process rank
# this is expected by pytorch to run distributed https://pytorch.org/docs/stable/elastic/run.html
try:
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl")
except KeyError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

# Load trained model and tokenizer
model_path = './data/ogt_protein_regressor/model/final/'
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
def tokenize_data(data):
    seqs = data["protein_seq"]
    return tokenizer(seqs, max_length=512, padding="max_length", truncation=True)
model = BertForSequenceClassification.from_pretrained(model_path).to(device)

# get the standardization parameters and define function to untransform labels
with open("./data/ogt_protein_regressor/data/standardization_params.json", "r") as file:
    standardization_params = json.load(file)
data_mean = torch.tensor(standardization_params['train_mean'], dtype=torch.float).to(device)
data_std = torch.tensor(standardization_params['train_std'], dtype=torch.float).to(device)
def unstandardize(predictions):
    return (torch.tensor(predictions).to(device)*data_std) + data_mean

# Ensure model is in evaluation mode
model.eval()

# Load and process test dataset
data_path = './data/ogt_protein_regressor/data'
data = load_from_disk(data_path)['test']#.select(range(1000))
data = data.map(prepare_aa_seq, batched=True)
data_encoded = data.map(tokenize_data, batched=True)
data_encoded.set_format('torch')
batches = DataLoader(data_encoded, batch_size=32)
with torch.no_grad():
    predictions = []
    targets = []
    for batch in batches:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Move the inputs to the same device as the model
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions.append(outputs.logits.cpu().numpy())
        targets.append(batch['labels'].cpu().numpy())

# Get true values and compute metrics
true_values = np.concatenate(targets).flatten()
predictions = np.concatenate(predictions).flatten()
predictions= unstandardize(predictions).cpu().numpy()
true_values = unstandardize(true_values).cpu().numpy()

mse = mean_squared_error(true_values, predictions)
r2 = r2_score(true_values, predictions)

print(f"MSE: {mse}, R^2: {r2}")

# Create a DataFrame for easy plotting
df = pd.DataFrame({
    'True Values': true_values,
    'Predictions': predictions,
})
df['error'] = df['Predictions'] - df['True Values']


# Plot distribution of true values vs predictions
plt.figure(figsize=(8, 8))
sns.distplot(df['True Values'], hist=False, label='True Values')
sns.distplot(df['Predictions'], hist=False, label='Predictions')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of True Values vs Predictions')
plt.legend()
plt.savefig('./data/ogt_protein_regressor/prediction_dist.png', dpi=200, bbox_inches='tight')

# Plot true values vs predictions
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x='True Values', y='Predictions', data=df, ax=ax)
ax.set_xlim(0, 105)
ax.set_ylim(0, 105)
ax.plot([0, 105], [0, 105], ls="--", c=".3")
ax.set_xlabel('True Values')
ax.set_ylabel('Predictions')
ax.set_title('True Values vs Predictions')
plt.savefig('./data/ogt_protein_regressor/prediction_scatter.png', dpi=200, bbox_inches='tight')

# and distribution of errors
plt.figure(figsize=(8, 8))
sns.distplot(df['error'], hist=False)
plt.xlabel('Error')
plt.ylabel('Density')
plt.savefig('./data/ogt_protein_regressor/error_dist.png', dpi=200, bbox_inches='tight')

# Define bins for 'True Values'
bins = pd.cut(df['True Values'], bins=10)
bins = bins.apply(lambda x: x.mid)

# Calculate mean and standard deviation of error as function of binned True Values
mean_error = df.groupby(bins)['error'].mean().reset_index()
std_error = df.groupby(bins)['error'].std().reset_index()

# Plot distribution of mean and std errors
plt.figure(figsize=(8, 8))
sns.lineplot(x=mean_error['True Values'], y=mean_error['error'], label='Mean Error')
plt.fill_between(mean_error['True Values'], 
                 mean_error['error'] - 1 * std_error['error'], 
                 mean_error['error'] + 1 * std_error['error'], 
                 color='b', alpha=0.2)
plt.xlabel('Binned True Values')
plt.ylabel('Error')
plt.title('Mean and Standard Deviation of Error vs Binned True Values')
plt.legend()
plt.savefig('./data/ogt_protein_regressor/error_bins.png', dpi=200, bbox_inches='tight')

# Compute the 95% confidence interval assuming Gaussian distribution
low_T, high_T = np.meshgrid(mean_error['True Values'], mean_error['True Values'])
dT = high_T - low_T
low_T_mean_error = np.interp(low_T, mean_error['True Values'], mean_error['error']).reshape(low_T.shape)
high_T_mean_error = np.interp(high_T, mean_error['True Values'], mean_error['error']).reshape(high_T.shape)
low_T_std_error = np.interp(low_T, std_error['True Values'], std_error['error']).reshape(low_T.shape)
high_T_std_error = np.interp(high_T, std_error['True Values'], std_error['error']).reshape(high_T.shape)

total_error_difference = np.abs(high_T_mean_error - 2*high_T_std_error) + np.abs(low_T_mean_error + 2*low_T_std_error) # 1 std on each side is 2 total, so 95%

df_error = pd.DataFrame({'dT': dT.flatten(), 'total_error_difference': total_error_difference.flatten()})
df_error = df_error[df_error['dT']>=0.0]

# Make the plot
fig, ax = plt.subplots(figsize=(8, 8))
sns.regplot(x='dT', y='total_error_difference', data=df_error, ax=ax)
plt.xlabel('Temperature Difference (C)')
plt.ylabel('Total Error Difference (C)')
plt.title('95% Confidence Interval of Error vs Temperature Difference')
ax.set_xlim(0, 105)
ax.set_ylim(0, 105)
ax.plot([0, 105], [0, 105], ls="--", c=".3")
plt.savefig('./data/ogt_protein_regressor/error_confidence.png', dpi=200, bbox_inches='tight')

# Compute probability of overlap assuming T distributions
import scipy.stats
df_overlap = pd.DataFrame(
    {'dT': dT.flatten(),
     'low_T_mean_error': low_T_mean_error.flatten(),
    'high_T_mean_error': high_T_mean_error.flatten(),
    'low_T_std_error': low_T_std_error.flatten(),
    'high_T_std_error': high_T_std_error.flatten()})

def solve(m1,m2,std1,std2):
  """Solve intersection of two gaussians"""
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

def do_one(row):
    """Compute the probability of overlap in error at some dT assuming gaussain error distributions.
    
    Eg. for some T1, the model makes predictions in a gaussain with mean T1+M1 and std S1, and for some T2=T1+dT,
    the model makes predictions in a gaussain with mean T2+M2 and std S2.
    We First we find the intersection point of the two gaussians, then we integrate the left gaussian from intersection to positive infinity
    and the right gaussian from intersection to negative infinity. The sum of these two integrals is the probability of obersving an overlap.
    """
    T1 = 0.0
    T2 = row['dT']
    M1 = row['low_T_mean_error']
    M2 = row['high_T_mean_error']
    S1 = row['low_T_std_error']
    S2 = row['high_T_std_error']
    try:
        intersections = solve(T1+M1,T2+M2,S1,S2)
        intersection= None
        for i in intersections:
            if i > T1+M1 and i < T2+M2:
                intersection = i
        if intersection is None:
            intersection = 1.0
    except:
        intersection = None
    # now integrate
    print(f"dT ({T2}), Left at {T1+M1} ({S1}), Right at {T2+M2}, ({S2}), Intersection {intersection}")
    if intersection is None:
        return None
    a1 = 1 - scipy.stats.norm.cdf(intersection, loc=T1+M1, scale=S1)
    a2 = scipy.stats.norm.cdf(intersection, loc=T2+M2, scale=S2)
    p = a1+a2
    print(f" Probability {p}")
    
    return p
    
df_overlap = df_overlap[df_overlap['dT']>0.0]
df_overlap['overlap_probability'] = df_overlap.apply(do_one, axis=1)
df_overlap = df_overlap.dropna()

# Make the plot
fig, ax = plt.subplots(figsize=(8, 8))
sns.regplot(x='dT', y='overlap_probability', data=df_overlap, logistic=True, ax=ax)
plt.xlabel('Temperature Difference (C)')
plt.ylabel('Overlap Probability')
plt.title('Probability of Error Overlap vs Temperature Difference')
plt.savefig('./data/ogt_protein_regressor/error_bounds.png', dpi=200, bbox_inches='tight')