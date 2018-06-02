"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def load_dataset_from_csv(path_csv, vocab):
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
        fields = tf.decode_csv(line, [None] * len(COLUMNS))
        labled_features = dict(zip(fields, COLUMNS)) # not used
        features = fields[:-1]
        label = fields[-1]
        return features, label

    dataset = dataset.map(_parse_line)

    # the dataset looks like this [({seq1:"...",seq2:[...],...}, {label:<0 or 1>}), ...]

    # Convert line into list of tokens, splitting by white space
    def _f1(s):
        tf.string_split(s)
    dataset = dataset.map(lambda x: ({k: _f1(v) for k, v in x[0].items()}, x[1]))

    # Lookup tokens to return their ids
    def _f2(tokens):
        (vocab.lookup(tokens), tf.size(tokens))
    dataset = dataset.map(lambda x: ({k: _f2(v) for k, v in x[0].items()}, x[1]))

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

    # I (pascal) find it unpractical to do the padding here...
    # Create batches and pad the sentences of different length
    # padded_shapes = ((tf.TensorShape([None]),  # sentence of unknown size
    #                   tf.TensorShape([])),     # size(words)
    #                  (tf.TensorShape([None]),  # labels of unknown size
    #                   tf.TensorShape([])))     # size(tags)

    # padding_values = ((params.id_pad_word,   # sentence padded on the right with id_pad_word
    #                    0),                   # size(words) -- unused
    #                   (params.id_pad_tag,    # labels padded on the right with id_pad_tag
    #                    0))                   # size(tags) -- unused


    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        # .padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    (stories, labels) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'story': story,
        'label': label,
        'iterator_init_op': init_op
    }

    return inputs
