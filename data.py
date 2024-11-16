import os.path as osp
import random
import torch
import zipfile
from joblib import Memory
from tqdm import tqdm

CACHE_DIR = '/kaggle/tmp/cache'
MEMORY = Memory(CACHE_DIR, verbose=2)
VALID_DATASETS = ['enron']

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_data_from_csv(csv_path, tokenizer, max_length=300, construct_textgraph=False, n_jobs=1,
                       force_lowercase=False, raw=False):
    df = pd.read_excel(csv_path)
    raw_documents = df['text'].tolist()
    labels = df['target'].tolist()

    N = len(raw_documents)
    train_indices, test_indices = train_test_split(range(N), test_size=0.2, random_state=42)

    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    if raw:
        return raw_documents, labels, train_mask, test_mask

    print(f"Encoding documents with max_length={max_length}...")
    docs = [tokenizer.encode(doc, truncation=True, max_length=300) for doc in raw_documents]

    label2index = {label: idx for idx, label in enumerate(set(labels))}
    label_ids = [label2index[label] for label in labels]

    if not construct_textgraph:
        return docs, label_ids, train_mask, test_mask, label2index


def shuffle_augment(docs: list, labels: list,
                    factor:float=1.0, random_seed=None):
    assert factor > 0.0
    if random_seed is not None:
        random.seed(random_seed)
    num_augment = int(len(docs) * factor)
    print(f"Generating {num_augment} augmented documents...")

    new_docs = []
    new_labels = []

    for __ in tqdm(range(num_augment)):
        idx = random.sample(range(len(docs)), k=1)[0]
        doc = docs[idx]
        perm_doc = random.sample(doc, k=len(doc))

        new_docs.append(perm_doc) 
        new_labels.append(labels[idx])

    return new_docs, new_labels
