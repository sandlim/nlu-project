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
from model.input_fn import input_fn
from model.input_fn import load_dataset_from_csv
from model.model_fn import model_fn

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir',
    default='experiments/base_model',
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
    params.eval_size = params.test_nlu18_size

    # Get paths for vocabularies and dataset
    path_vocab = os.path.join(args.data_dir, 'vocab.txt')
    path_test_stories1 = os.path.join(args.data_dir, 'test_nlu18/stories1.csv')
    path_test_stories2 = os.path.join(args.data_dir, 'test_nlu18/stories2.csv')

    # Load Vocabularies
    vocab = tf.contrib.lookup.index_table_from_file(
        path_vocab, num_oov_buckets=num_oov_buckets)

    vocab_back = open(path_vocab, 'r').read().splitlines()

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_stories1 = load_dataset_from_csv(path_test_stories1)
    test_stories2 = load_dataset_from_csv(path_test_stories2)

    # Specify other parameters for the dataset and the model
    params.eval_size = params.test_nlu18_size
    params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))

    inputs = input_fn('eval', [test_stories1, test_stories2], vocab, params)

    logging.info("- done.")

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', inputs, params)
    logging.info("- done.")

    logging.info("Starting inference")
    preds = infer(model_spec, args.model_dir, params, args.restore_from)

    save_path = os.path.join(args.data_dir, 'test_nlu18', 'predictions.csv')
    print("Saving predictions in {}...".format(save_path))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, 'w') as f:
        for pred in preds:
            for p in pred:
                f.write(str(p + 1) + '\n')

    # Export the dataset
    print("- done.")
