import csv
import itertools as it
import logging

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Memory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AdamW, AutoTokenizer, BertConfig,
                          BertForSequenceClassification, BertModel,
                          BertTokenizer, DistilBertConfig,
                          DistilBertForSequenceClassification, DistilBertModel,
                          DistilBertTokenizer, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaModel)

from sklearn.metrics import f1_score
from data import shuffle_augment, load_data_from_csv


try:
    import wandb
    WANDB = True
except ImportError:
    print("WandB not installed, to track experiments: pip install wandb")
    WANDB = False


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()
CACHE_DIR = '/kaggle/tmp/cache/textclf'
MEMORY = Memory(CACHE_DIR, verbose=2)

VALID_DATASETS = ['enron']

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertModel),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaModel),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel)
}
def pad(seqs, with_token=0, to_length=None):
    if to_length is None:
        to_length = max(len(seq) for seq in seqs)
    return [seq + (to_length - len(seq)) * [with_token] for seq in seqs]


def get_collate_for_transformer(pad_token_id):
    def _collate_for_transformer(examples):
        docs, labels = list(zip(*examples))
        input_ids = torch.tensor(pad(docs, with_token=pad_token_id))
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        attention_mask[input_ids == pad_token_id] = 0
        labels = torch.tensor(labels)
        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, attention_mask, token_type_ids, labels
    return _collate_for_transformer


def train(args,  train_data, model, tokenizer):
    collate_fn = get_collate_for_transformer(tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=('cuda' in str(args.device)))

    t_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    log_dir = '/kaggle/working/runs' 
    writer = SummaryWriter(log_dir=log_dir)

    if args.ignore_position_ids:
        print("Setting position ids to zero and ignoring grad")
        if args.model_type == 'bert':
            model.bert.embeddings.position_embeddings.weight.requires_grad = False
            model.bert.embeddings.position_embeddings.weight.zero_()
        else:
            raise NotImplementedError("Ignore position ids only implemented for BERT")

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size  = %d", args.batch_size)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.epochs, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
            if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  
            if args.ignore_position_ids:
                    inputs['position_ids'] = torch.zeros(
                        inputs['input_ids'].shape[0],  
                        inputs['input_ids'].shape[1],  
                        device=inputs['input_ids'].device,
                        dtype=torch.long
                    )
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if WANDB:
                    wandb.log({'epoch': epoch,
                               'lr': scheduler.get_last_lr()[0],
                               'loss': loss})

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:

                avg_loss = (tr_loss - logging_loss)/ args.logging_steps
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss', avg_loss, global_step)
                logging_loss = tr_loss

    writer.close()
    return global_step, tr_loss / global_step

import pandas as pd

def evaluate(args, dev_or_test_data, model, tokenizer):

    collate_fn = get_collate_for_transformer(tokenizer.pad_token_id)
    
    data_loader = torch.utils.data.DataLoader(dev_or_test_data,
                                              collate_fn=collate_fn,
                                              num_workers=args.num_workers,
                                              batch_size=args.test_batch_size,
                                              pin_memory=('cuda' in str(args.device)),
                                              shuffle=False)
    
    all_logits = []
    all_targets = []
    all_texts = []  
    nb_eval_steps, eval_loss = 0, 0.0
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        original_texts = batch[-1] 
        
        with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert'] else None
                outputs = model(**inputs)
                all_targets.append(inputs['labels'].detach().cpu())
        
        nb_eval_steps += 1
        loss, logits = outputs[:2]
        eval_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu())
        
        all_texts.extend(original_texts) 
    
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    
    eval_loss /= nb_eval_steps
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets).sum() / targets.size

    f1_micro = f1_score(targets, preds, average='micro')
    f1_macro = f1_score(targets, preds, average='macro', zero_division=1)

    if WANDB:
        wandb.log({"test/acc": acc, "test/loss": eval_loss,
                   "test/f1_micro": f1_micro,
                   "test/f1_macro": f1_macro})
                
    print("acc\n:", acc)
    print("f1_micro\n:", f1_micro)
    print("f1_macro\n:", f1_macro)

    return acc, eval_loss


def run_xy_model(args):
    print("Loading data...")
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    embedding = None
    print("Using tokenizer:", tokenizer)

    do_truncate = not (args.stats_and_exit or args.model_type == 'mlp')
    if args.stats_and_exit:
        max_length = None
    else:
        max_length = 1024 

    enc_docs, enc_labels, train_mask, test_mask, label2index = load_data_from_csv('/kaggle/input/d/aks512345/email-classification/text-clf-baselines-main/text-clf-baselines-main/data/enron_filtered4.xlsx',
                                                                         tokenizer,
                                                                         max_length=max_length,
                                                                         construct_textgraph=False,
                                                                         n_jobs=args.num_workers)
    print("Done")

    lens = np.array([len(doc) for doc in enc_docs])
    print("Min/max document length:", (lens.min(), lens.max()))
    print("Mean document length: {:.4f} ({:.4f})".format(lens.mean(), lens.std()))
    assert len(enc_docs) == len(enc_labels) == train_mask.size(0) == test_mask.size(0)
    enc_docs_arr, enc_labels_arr = np.array(enc_docs, dtype='object'), np.array(enc_labels)

    train_docs = enc_docs_arr[train_mask]
    train_labels = enc_labels_arr[train_mask]

    if args.shuffle_augment:
        factor = float(args.shuffle_augment)

        new_docs, new_labels = shuffle_augment(list(train_docs),
                                               list(train_labels),
                                               factor=factor,
                                               random_seed=args.seed)

        new_docs = np.array(new_docs, dtype='object')
        new_labels = np.array(new_labels)

        train_docs = np.concatenate([train_docs, new_docs])
        train_labels = np.concatenate([train_labels, new_labels])

    train_data = list(zip(train_docs, train_labels))

    test_data = list(zip(enc_docs_arr[test_mask], enc_labels_arr[test_mask]))

    print("N", len(enc_docs))
    print("N train", len(train_data))
    print("N test", len(test_data))
    print("N classes", len(label2index))

    if args.stats_and_exit:
        print("Warning: length stats depend on tokenizer and max_length of model")
        exit(0)


    config_class, model_class, __ = MODEL_CLASSES[args.model_type]
    print("Loading", args.model_type)
    print("Loading config")
    config = config_class.from_pretrained(args.model_name_or_path,
                                              num_labels=len(label2index),
                                              cache_dir=CACHE_DIR)

    print(config)
    print("Loading model")
    model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=CACHE_DIR)

    model.to(args.device)


    if WANDB:
        wandb.watch(model, log_freq=args.logging_steps)


    train(args, train_data, model, tokenizer)
    acc, eval_loss = evaluate(args, test_data, model, tokenizer)
    print(f"[{args.dataset}] Test accuracy: {acc:.4f}, Eval loss: {eval_loss}")
    return acc


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', choices=VALID_DATASETS)
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        choices=["distilbert", "bert", "roberta"])
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--results_file", default=None,
                        help="Store results to this results file")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=None,
                        help="Batch size for testing (defaults to train batch size)")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    parser.add_argument('--unfreeze_embedding', dest="freeze_embedding", default=True,
            action='store_false', help="Allow updating pretrained embeddings")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers")

    parser.add_argument("--stats_and_exit", default=False,
                        action='store_true',
                        help="Print dataset stats and exit.")

    parser.add_argument("--ignore_position_ids",
                        help="Use all zeros to pos ids",
                        default=False, action='store_true')
    parser.add_argument("--seed", default=None,
                        help="Random seed for shuffle augment")
    parser.add_argument("--shuffle_augment", type=float,
                        default=0, help="Factor for shuffle data augmentation")
    ##########################
    args = parser.parse_args()


    assert args.model_name_or_path, f"Please supply --model_name_or_path for {args.model_type}"

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.test_batch_size = args.batch_size if args.test_batch_size is None else args.test_batch_size

    if WANDB:
        wandb.init(project="text-clf")
        wandb.config.update(args)

    acc = {
        'bert': run_xy_model,
        'distilbert': run_xy_model,
        'roberta': run_xy_model
    }[args.model_type](args)
    if args.results_file:
        with open(args.results_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([args.model_type,args.dataset,acc])


if __name__ == '__main__':
    main()
