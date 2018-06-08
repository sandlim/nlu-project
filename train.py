"""Train the model"""

import argparse
import logging
import os, shutil

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
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
    '--restore_dir',
    default=None,
    help="Optional, directory containing weights to reload before training")
parser.add_argument(
    '--overwrite', dest='overwrite', default=False, action='store_true')
parser.add_argument(
    '--use_shuffled', dest='use_shuffled', default=False, action='store_true')
parser.add_argument(
    '--use_antonym_simple', dest='use_antonym_simple', default=False, action='store_true')
parser.add_argument(
    '--use_antonym_nltk', dest='use_antonym_nltk', default=False, action='store_true')
parser.add_argument(
    '--use_seq2seq', dest='use_seq2seq', default=False, action='store_true')
parser.add_argument(
    '--use_dev', dest='use_dev', default=False, action='store_true')

if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(
        json_path), "No json file found at {}, run build_vocab.py".format(
            json_path)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets  # number of buckets for unknown words

    # Check that we are not overwriting some previous experiment
    if args.overwrite:
        if os.path.isdir(os.path.join(args.model_dir, 'best_weights')):
            shutil.rmtree(os.path.join(args.model_dir, 'best_weights'))
        if os.path.isdir(os.path.join(args.model_dir, 'eval_summaries')):
            shutil.rmtree(os.path.join(args.model_dir, 'eval_summaries'))
        if os.path.isdir(os.path.join(args.model_dir, 'last_weights')):
            shutil.rmtree(os.path.join(args.model_dir, 'last_weights'))
        if os.path.isdir(os.path.join(args.model_dir, 'train_summaries')):
            shutil.rmtree(os.path.join(args.model_dir, 'train_summaries'))

    model_dir_has_best_weights = os.path.isdir(
        os.path.join(args.model_dir, "best_weights"))
    overwriting = model_dir_has_best_weights and args.restore_dir is None
    assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    gpu_options = tf.GPUOptions(allow_growth=True)
    params.sessConfig = tf.ConfigProto(gpu_options=gpu_options)
    params.data_dir = args.data_dir

    # Get paths for vocabularies and dataset
    path_vocab = os.path.join(args.data_dir, 'vocab.txt')
    path_train_stories = os.path.join(args.data_dir, 'train/stories.csv')
    path_shuffled_endings = os.path.join(args.data_dir, 'train/shuffled_endings.csv')
    path_antonym_endings_simple = os.path.join(args.data_dir, 'train/antonym_endings_simple.csv')
    path_antonym_endings_nltk = os.path.join(args.data_dir, 'train/antonym_endings_nltk.csv')
    path_seq2seq_endings = os.path.join(args.data_dir, 'train/seq2seq_endings.csv')
    path_dev_stories_c = os.path.join(args.data_dir, 'dev/stories1.csv')
    path_dev_stories_w = os.path.join(args.data_dir, 'dev/stories2.csv')
    path_val_stories_c = os.path.join(args.data_dir, 'val/stories1.csv')
    path_val_stories_w = os.path.join(args.data_dir, 'val/stories2.csv')

    # Load Vocabularies
    vocab = tf.contrib.lookup.index_table_from_file(
        path_vocab, num_oov_buckets=num_oov_buckets)

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    train_stories = load_dataset_from_csv(path_train_stories)
    shuffled_endings = load_dataset_from_csv(path_shuffled_endings)
    antonym_endings_simple = load_dataset_from_csv(path_antonym_endings_simple)
    antonym_endings_nltk = load_dataset_from_csv(path_antonym_endings_nltk)
    seq2seq_endings = load_dataset_from_csv(path_seq2seq_endings)
    dev_stories1 = load_dataset_from_csv(path_dev_stories_c)
    dev_stories2 = load_dataset_from_csv(path_dev_stories_w)
    val_stories1 = load_dataset_from_csv(path_val_stories_c)
    val_stories2 = load_dataset_from_csv(path_val_stories_w)

    if args.use_shuffled:
        params.train_size += sum(1 for line in open(path_shuffled_endings)) - 1
        train_stories = train_stories.concatenate(shuffled_endings)
    if args.use_antonym_simple:
        params.train_size += sum(1 for line in open(path_antonym_endings_simple)) - 1
        train_stories = train_stories.concatenate(antonym_endings_simple)
    if args.use_antonym_nltk:
        params.train_size += sum(1 for line in open(path_antonym_endings_nltk)) - 1
        train_stories = train_stories.concatenate(antonym_endings_nltk)
    if args.use_seq2seq:
        params.train_size += sum(1 for line in open(path_seq2seq_endings)) - 1
        train_stories = train_stories.concatenate(seq2seq_endings)

    if args.use_dev:
        params.train_size += params.dev_size * 2
    else:
        params.eval_size += params.dev_size

    # Specify other parameters for the dataset and the model
    params.buffer_size = params.train_size  # buffer size for shuffling
    params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))

    if args.use_dev:
        train_inputs = input_fn('train_including_dev',
                                [train_stories, dev_stories1, dev_stories2],
                                vocab, params)
        val_inputs = input_fn(
            'eval', [val_stories1, val_stories2],
            vocab, params)
    else:
        train_inputs = input_fn('train', [train_stories], vocab, params)
        val_inputs = input_fn('eval', [
            val_stories1.concatenate(dev_stories1),
            val_stories2.concatenate(dev_stories2)
        ], vocab, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    logging.info("- created train-model...")
    eval_model_spec = model_fn('eval', val_inputs, params, reuse=True)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir,
                       params, args.restore_dir)
