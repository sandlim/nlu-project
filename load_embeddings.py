from gensim import models
import tensorflow as tf
import numpy as np
import os

def load_embedding(session, params):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''

    emb, dim_embedding, vocab_size = params.embeddings, params.embedding_size, params.vocab_size

    path = os.path.join(params.data_dir, '..', 'glove.6B', 'glove.6B.100d.word2vec.txt')

    vocab_path = os.path.join(params.data_dir, 'vocab.txt')

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for idx, tok in enumerate(open(vocab_path, 'r').read().splitlines()):
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding}) # here, embeddings are actually set