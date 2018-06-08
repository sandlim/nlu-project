"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.inference import infer
from model.generator_input_fn import input_fn
from model.generator_input_fn import load_dataset_from_csv
from model.generator_model_fn import model_fn

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir',
    default='experiments/generator_model',
    help="Directory containing params.json")
parser.add_argument(
    '--data_dir',
    default='data/dev_split',
    help="Directory containing the dataset")
parser.add_argument(
    '--restore_from',
    default='best_weights',
    help="Subdirectory of model dir or file containing the weights")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(
        json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets  # number of buckets for unknown words

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    gpu_options = tf.GPUOptions(allow_growth=True)
    params.sessConfig = tf.ConfigProto(gpu_options=gpu_options)
    params.data_dir = args.data_dir

    # Get paths for vocabularies and dataset
    path_vocab = os.path.join(args.data_dir, 'vocab.txt')
    path_train_stories = os.path.join(args.data_dir, 'train/stories.csv')

    # Load Vocabularies
    vocab = tf.contrib.lookup.index_table_from_file(
        path_vocab, num_oov_buckets=num_oov_buckets)

    vocab_back = open(path_vocab, 'r').read().splitlines()

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    train_stories = load_dataset_from_csv(path_train_stories)

    # Specify other parameters for the dataset and the model
    params.eval_size = params.train_size
    params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))

    inputs = input_fn('infer', [train_stories], vocab, params)

logging.info("- done.")

# Define the model
logging.info("Creating the model...")
model_spec = model_fn('infer', inputs, params)
logging.info("- done.")

logging.info("Starting inference")
pred = infer(model_spec, args.model_dir, params, args.restore_from)
df = pd.read_csv(
            path_train_stories,
            usecols=['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5', 'label'])

stringify = np.vectorize(lambda x: vocab_back[x])
string_pred = stringify(pred)
df['sentence5'] = [' '.join(p) for p in string_pred]
df['label'] = 0

save_path = os.path.join(args.data_dir, 'train', 'seq2seq_endings.csv')
print("Saving predictions in {}...".format(save_path))
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

# Export the dataset
df.to_csv(save_path, index=False)
print("- done.")
