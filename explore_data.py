"""Quick exploration script to understand the dataset."""
import pandas as pd
import json

# Load data
train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/validation.csv')
test = pd.read_csv('data/processed/test.csv')

print('=== DATASET OVERVIEW ===')
print(f'Train: {len(train)} samples')
print(f'Validation: {len(val)} samples')
print(f'Test: {len(test)} samples')
print(f'Total: {len(train)+len(val)+len(test)} samples')

print('\n=== LABEL DISTRIBUTION (Train) ===')
print(train['label'].value_counts().sort_index())

print('\n=== LABEL DISTRIBUTION (Test) ===')
print(test['label'].value_counts().sort_index())

print('\n=== TEXT LENGTH STATS (word count) ===')
train['word_count'] = train['text'].str.split().str.len()
print(f'Mean: {train["word_count"].mean():.1f}')
print(f'Std: {train["word_count"].std():.1f}')
print(f'Min: {train["word_count"].min()}')
print(f'Max: {train["word_count"].max()}')
print(f'Median: {train["word_count"].median():.1f}')
print(f'95th percentile: {train["word_count"].quantile(0.95):.0f}')
print(f'99th percentile: {train["word_count"].quantile(0.99):.0f}')

print('\n=== TEXT LENGTH BY INTENT ===')
for intent in sorted(train['label'].unique()):
    subset = train[train['label']==intent]['word_count']
    print(f'{intent:25s}: mean={subset.mean():.1f}, max={subset.max()}, std={subset.std():.1f}')

print('\n=== CHAR LENGTH STATS ===')
train['char_count'] = train['text'].str.len()
print(f'Mean chars: {train["char_count"].mean():.1f}')
print(f'Max chars: {train["char_count"].max()}')
print(f'95th pct chars: {train["char_count"].quantile(0.95):.0f}')

print('\n=== SAMPLE TEXTS PER INTENT ===')
for intent in sorted(train['label'].unique()):
    samples = train[train['label']==intent]['text'].head(3).tolist()
    print(f'\n{intent}:')
    for s in samples:
        print(f'  "{s}"')

print('\n=== TOKENIZER ANALYSIS ===')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
token_lengths = train['text'].apply(lambda x: len(tokenizer.encode(x)))
print(f'Mean tokens: {token_lengths.mean():.1f}')
print(f'Max tokens: {token_lengths.max()}')
print(f'95th pct tokens: {token_lengths.quantile(0.95):.0f}')
print(f'99th pct tokens: {token_lengths.quantile(0.99):.0f}')
print(f'Percent under 32 tokens: {(token_lengths <= 32).mean()*100:.1f}%')
print(f'Percent under 48 tokens: {(token_lengths <= 48).mean()*100:.1f}%')
print(f'Percent under 64 tokens: {(token_lengths <= 64).mean()*100:.1f}%')
