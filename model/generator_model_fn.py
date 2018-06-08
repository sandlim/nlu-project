"""Define the model."""

import tensorflow as tf


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

    story = inputs['story']

    # Get word embeddings for each token in the sentence
    embeddings = tf.get_variable(
        name="embeddings",
        dtype=tf.float32,
        shape=[params.vocab_size, params.embedding_size], trainable=False)
    params.embeddings = embeddings

    length = [
        s[1] for k, s in story.items()
    ]
    story = [
        tf.nn.embedding_lookup(params.embeddings, s[0]) for k, s in story.items()
    ]
    # story[0] = tf.Print(story[0], [story[0], tf.shape(story[0])], message='story[0]')
    # story[1] = tf.Print(story[1], [story[1], tf.shape(story[1])], message='story[1]')
    # Apply LSTM over the embeddings
    lstm_cell_beg = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        lstm_cell_beg, story[0], sequence_length=length[0], dtype=tf.float32)

    projection_layer = tf.layers.Dense(params.vocab_size, use_bias=False)
    lstm_cell_end = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
    if mode != 'infer':
        helper = tf.contrib.seq2seq.TrainingHelper(story[1], length[1])
        decoder = tf.contrib.seq2seq.BasicDecoder(
            lstm_cell_end, helper, encoder_state,
            output_layer=projection_layer)
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output
    else:
        print(encoder_state)
        # Helper
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embeddings,
            tf.fill([tf.shape(encoder_outputs)[0]], 0), 13)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            lstm_cell_end, helper, encoder_state,
            output_layer=projection_layer)
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=30)
        translations = outputs.sample_id

    return outputs


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
    label = tf.pad(inputs['story']['end'][0][:, 1:], [[0, 0], [0, 1]])
    # sentence_lengths = tf.stack(
    #     [tf.squeeze(s[1]) for k, s in inputs['story'].items()])

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions

        outputs = build_model(mode, inputs, params)

    if mode == 'infer':
        prediction = outputs.sample_id
    else:
        logits = outputs.rnn_output
        prediction = tf.argmax(logits, -1)


        # Define loss and accuracy (we need to apply a mask to account for padding)
        raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label)
        length = [
            s[1] for k, s in inputs['story'].items()
        ]
        mask = tf.sequence_mask(length[1])
        losses = tf.boolean_mask(raw_loss, mask)
        loss = tf.reduce_mean(losses)
        perplexity = tf.exp(loss)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(label, prediction), tf.float32), mask))

        # Define training step that minimizes the loss with the Adam optimizer
        if is_training:
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
            global_step = tf.train.get_or_create_global_step()
            # train_op = optimizer.minimize(loss, global_step=global_step)
            # clipping
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        with tf.variable_scope("metrics"):
            metrics = {
                'accuracy':
                tf.metrics.mean(accuracy),
                'perplexity':
                tf.metrics.mean(tf.exp(loss)),
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
    if mode != 'infer':
        model_spec['loss'] = loss
        model_spec['accuracy'] = accuracy
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec

def my_print(t, m):
    t = tf.Print(t, [t, tf.shape(t)], message=m)
    return t
