"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def load_dataset_from_csv(path_csv, vocab, params):
    """Create tf.data Instance from csv file

    Args:
        path_csv: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load csv file, one example per line
    dataset = tf.data.TextLineDataset(path_csv).skip(1)

    def _parse_line(line):
        COLUMNS = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5','label']
        fields = tf.decode_csv(line, [tf.constant([""])] * len(COLUMNS))
        if params.concat_first_four == True:
            features = {'beg':tf.expand_dims(fields[0] + fields[1] + fields[2] + fields[3], 0), 'end':tf.expand_dims(fields[4], 0)}
        else:
            features =  dict(zip(COLUMNS[:-1], fields[:-1]))
        label = fields[-1]
        return features, label

    dataset = dataset.map(_parse_line)
    print(dataset)
    # the dataset looks like this [([seq1,...,seq5],label),(...,...),...]
    # seqi and label are tensors

    # Convert line into list of tokens, splitting by white space
    def _split(s):
        return tf.string_split(s).values
    dataset = dataset.map(lambda x1, x2: ({k:_split(s) for k,s in x1.items()}, x2))

    # Lookup tokens to return their ids
    def _vocabularize(tokens):
        return (vocab.lookup(tokens), tf.size(tokens))
    dataset = dataset.map(lambda x1, x2: ({k:_vocabularize(s) for k,s in x1.items()}, x2))

    return dataset


def input_fn(mode, datasets, params):
    """Input function

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        datasets: (list of tf.Dataset)
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train' or mode == 'train_including_dev')
    buffer_size = params.buffer_size if is_training else 1

    # Zip the sentence and the labels together
    dataset = datasets[0]
    if mode == 'train_including_dev':
        dataset = dataset.concatenate(datasets[1])

    # Create batches and pad the sentences of different length
    if params.concat_first_four:
        padded_shapes = ({'beg':(tf.TensorShape([None]),  # sentence 1 - 4 of unknown size
                           tf.TensorShape([])),     # sequence_length
                          'end':(tf.TensorShape([None]),  # sentence 5 of unknown size
                           tf.TensorShape([]))},    # sequence_length
                          tf.TensorShape([]))       # label
    else: 
        padded_shapes = ([(tf.TensorShape([None]),  # sentence 1 of unknown size
                          tf.TensorShape([])),      # sequence_length
                         (tf.TensorShape([None]),   # sentence 2 of unknown size
                          tf.TensorShape([])),     # sequence_length
                         (tf.TensorShape([None]),   # sentence 3 of unknown size
                          tf.TensorShape([])),     # sequence_length
                         (tf.TensorShape([None]),   # sentence 4 of unknown size
                          tf.TensorShape([])),     # sequence_length
                         (tf.TensorShape([None]),   # sentence 5 of unknown size
                          tf.TensorShape([]))],     # sequence_length
                          tf.TensorShape([]))       # label

    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .padded_batch(params.batch_size, padded_shapes=padded_shapes)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    (stories, labels) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'stories': stories,
        'labels': labels,
        'iterator_init_op': init_op
    }

    return inputs
