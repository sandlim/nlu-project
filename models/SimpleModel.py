import os, copy
import tensorflow as tf
from models.BasicModel import BasicModel


class SimpleModel(BasicModel):

    def set_model_props(self):
        # This function is here to be overriden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def build_graph(self, graph):
        train_data = tf.data.TextLineDataset(self.config['train_data_path']).skip(1)
        val_data = tf.data.TextLineDataset(self.config['val_data_path']).skip(1)

        def _parse_line(line, train=True):
            if train:
                COLUMNS = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
            else:
                COLUMNS = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'ending1', 'ending2', 'rightEnding']
            # Decode the line into its fields
            num_cols = 7 if train else 8
            fields = tf.decode_csv(line, [None] * num_cols)

            # Pack the result into a dictionary
            features = dict(zip(COLUMNS, fields[2:]))

            return features

        train_data.map(_parse_line, train=True)
        val_data.map(_parse_line, train=False)

    def infer(self):
        raise Exception('The infer function must be overriden by the model')

    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the model')

    def train(self, save_every=1):
        # This function is usually common to all your models, Here is an example:
        for epoch_id in range(0, self.max_iter):
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and epoch_id % save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, epoch_id = self.sess.run([global_step_t, self.epoch_id])
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(epoch_id), global_step)

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:

            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def infer(self):
        # This function is usually common to all your models
        pass
