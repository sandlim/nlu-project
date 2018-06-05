"""Define the model."""

import tensorflow as tf

def rnn_logits(story, params):
    # Get word embeddings for each token in the sentence
    embeddings = tf.get_variable(
        name="embeddings",
        dtype=tf.float32,
        shape=[params.vocab_size, params.embedding_size])
    story = [
        tf.nn.embedding_lookup(embeddings, s[0]) for k, s in story.items()
    ]
    # Apply LSTM over the embeddings
    with tf.variable_scope('lstm-beg'):
        lstm_cell_beg = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        output_beg, _ = tf.nn.dynamic_rnn(
            lstm_cell_beg, story[0], dtype=tf.float32)
    with tf.variable_scope('lstm-end'):
        lstm_cell_end = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        output_end, _ = tf.nn.dynamic_rnn(
            lstm_cell_end, story[1], dtype=tf.float32)

    lstm_output = tf.concat([output_beg, output_end], axis=1)
    with tf.variable_scope('H_layer'):
        output = tf.layers.dense(lstm_output, 256, name='H_output')
        # TODO
        # output = tf.layers.dense(lstm_output, params.H_size)

    # Compute logits from the output of the LSTM
    logits = tf.layers.dense(output, 2, name='rnn_logits')
    return logits


def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    if mode == 'train':
        story = inputs['story']
    elif mode == 'eval':
        story_c = inputs['story_c']
        story_w = inputs['story_w']

    if params.model_version == 'lstm':
        if mode == 'train':
            logits = rnn_logits(story, params)
        elif mode == 'eval':
            logits_c = rnn_logits(story_c, params)
            logits_w = rnn_logits(story_w, params)
            logits = tf.stack([logits_c[1], logits_w[1]])

    else:
        raise NotImplementedError("Unknown model version: {}".format(
            params.model_version))

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    label = tf.expand_dims(inputs['label'], axis=0)
    # sentence_lengths = tf.stack(
    #     [tf.squeeze(s[1]) for k, s in inputs['story'].items()])

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params)
        prediction = tf.cast(tf.argmax(logits, 0), tf.int32)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=label)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(label, prediction), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy':
            tf.metrics.accuracy(labels=label, predictions=prediction),
            'loss':
            tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(
        *[tf.global_variables_initializer(),
          tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = prediction
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
