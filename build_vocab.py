"""Build vocabularies of words and tags from datasets"""

import argparse
from collections import Counter
import json
import os
import sys
import pandas as pd
import re


parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=3, help="Minimum count for words in the dataset",
                    type=int)
parser.add_argument('--data_dir', default='data/dev_split', help="Directory containing the dataset")

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '<pad>'


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(csv_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        for i in range(5):
            sentence = row["sentence%d" % (i + 1)]
            vocab.update(sentence.strip().split(' '))

    return len(df)


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    counter = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/stories.csv'), counter)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'dev/stories1.csv'), counter)
    update_vocab(os.path.join(args.data_dir, 'dev/stories2.csv'), counter)
    size_val_sentences = update_vocab(os.path.join(args.data_dir, 'val/stories1.csv'), counter)
    update_vocab(os.path.join(args.data_dir, 'val/stories2.csv'), counter)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/stories1.csv'), counter)
    update_vocab(os.path.join(args.data_dir, 'test/stories2.csv'), counter)
    print("- done.")

    # Only keep most frequent tokens
    vocab = [tok for tok, count in counter.items() if count >= args.min_count_word]

    # Add pad tokens
    if PAD_WORD not in vocab:
        vocab = [PAD_WORD] + vocab

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(vocab, os.path.join(args.data_dir, 'vocab.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'eval_size': size_val_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(vocab) + NUM_OOV_BUCKETS,
        'pad_word': PAD_WORD,
        'num_oov_buckets': NUM_OOV_BUCKETS
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
