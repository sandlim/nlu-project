"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def load_dataset_from_csv(path_csv):
    """Create tf.data Instance from csv file

    Args:
        path_csv: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load csv file, one example per line
    return tf.data.TextLineDataset(path_csv).skip(1)


def process_text_line_dataset(dataset, mode, vocab, params):

    def _parse_line(line):
        COLUMNS = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'label']
        fields = tf.decode_csv(
            line,
            ([tf.constant([""])] * (len(COLUMNS) - 1)) + [tf.constant([0])])
        if params.concat_first_four == True:
            features = {
                'beg':
                tf.expand_dims(
                    tf.string_join(
                        [fields[0], fields[1], fields[2], fields[3]],
                        separator=' ',
                        name=None), 0),
                'end':
                tf.expand_dims(fields[4], 0)
            }
        else:
            features = dict(zip(COLUMNS[:-1], fields[:-1]))
        label = fields[-1]
        return features, label

    if mode != 'infer':
        dataset = dataset.map(lambda l1, l2: (_parse_line(l1), _parse_line(l2)))
    else:
        dataset = dataset.map(_parse_line)

    # the dataset looks like this [([seq1,...,seq5],label),(...,...),...]
    # seqi and label are tensors

    # Convert line into list of tokens, splitting by white space
    def _split(s):
        return tf.string_split(s).values

    # Lookup tokens to return their ids
    def _vocabularize(tokens):
        return (vocab.lookup(tokens), tf.size(tokens))

    apply_split = lambda x1, x2: ({k: _split(s) for k, s in x1.items()}, x2)
    apply_vocabularize = lambda x1, x2: ({k: _vocabularize(s) for k, s in x1.items()}, x2)

    if mode != 'infer':
        dataset = dataset.map(lambda l1, l2: (apply_split(*l1), apply_split(*l2)))
        dataset = dataset.map(lambda l1, l2: (apply_vocabularize(*l1), apply_vocabularize(*l2)))
    else:
        dataset = dataset.map(apply_split)
        dataset = dataset.map(apply_vocabularize)

    return dataset


def input_fn(mode, datasets, vocab, params):
    """Input function

    Args:
        :param mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        :param datasets: (list of tf.Dataset)
        :param vocab: The vocab
        :param params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train' or mode == 'train_including_dev')
    buffer_size = params.buffer_size if is_training else 1

    # Zip the sentence and the labels together
    dataset = datasets[0]

    if mode != 'infer':
        dataset = tf.data.Dataset.zip((dataset, datasets[1]))
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = process_text_line_dataset(dataset, mode, vocab, params)

    if mode != 'infer':
        dataset = dataset.map(lambda l1, l2: tf.cond(tf.squeeze(tf.equal(l1[1], tf.constant([0]))), lambda: l1, lambda: l2))

    # Create batches and pad the sentences of different length
    padded_shapes = (
        {
            'beg': (
                tf.TensorShape([None]),  # sentence 1 - 4 of unknown size
                tf.TensorShape([])),  # sequence_length
            'end': (
                tf.TensorShape([None]),  # sentence 5 of unknown size
                tf.TensorShape([]))  # sequence_length
        },
        tf.TensorShape([]))  # label

    dataset = (
        dataset.padded_batch(
            params.batch_size, padded_shapes=padded_shapes)
        .prefetch(2)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    init_op = iterator.initializer

    # Build and return a dictionary containing the nodes / ops
    (story, label) = iterator.get_next()
    inputs = {'story': story, 'label': label, 'iterator_init_op': init_op}

    return inputs
