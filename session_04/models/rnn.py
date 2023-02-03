import numpy as np
import tensorflow as tf_v2
tf_v2.compat.v1.disable_v2_behavior()


MAX_SENTENCE_LENGTH = 500
NUM_CLASSES = 20

class RNN:
  def __init__(self,
               vocab_size,
               embedding_size,
               lstm_size,
               batch_size):
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._lstm_size = lstm_size
    self._batch_size = batch_size

    self._data = tf_v2.compat.v1.placeholder(tf_v2.compat.v1.int32, shape=[batch_size, MAX_SENTENCE_LENGTH])
    self._labels = tf_v2.compat.v1.placeholder(tf_v2.compat.v1.int32, shape=[batch_size, ])
    self._sentence_lengths = tf_v2.compat.v1.placeholder(tf_v2.compat.v1.int32, shape=[batch_size, ])
    self._final_tokens = tf_v2.compat.v1.placeholder(tf_v2.compat.v1.int32, shape=[batch_size, ])


  def embedding_layer(self, indices):
    pretrained_vectors = []
    pretrained_vectors.append(np.zeros(self._embedding_size))
    np.random.seed(2021)
    for _ in range (self._vocab_size + 1):
      pretrained_vectors.append(np.random.normal(loc=0., scale=1., size=self._embedding_size))

    pretrained_vectors = np.array(pretrained_vectors)

    self._embedding_matrix = tf_v2.compat.v1.get_variable(
        name='embedding',
        shape=(self._vocab_size + 2, self._embedding_size),
        initializer=tf_v2.compat.v1.constant_initializer(pretrained_vectors)
    )
    return tf_v2.compat.v1.nn.embedding_lookup(self._embedding_matrix, indices)


  def LSTM_layer(self, embeddings):
    lstm_cell = tf_v2.compat.v1.nn.rnn_cell.BasicLSTMCell(self._lstm_size)
    zero_state = tf_v2.compat.v1.zeros(shape=(self._batch_size, self._lstm_size))
    initial_state = tf_v2.compat.v1.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state)

    lstm_inputs = tf_v2.compat.v1.unstack(
        tf_v2.compat.v1.transpose(embeddings, perm=[1, 0, 2])
    )
    lstm_outputs, last_state = tf_v2.compat.v1.nn.static_rnn(
        cell=lstm_cell,
        inputs=lstm_inputs,
        initial_state=initial_state,
        sequence_length=self._sentence_lengths
    )
    lstm_outputs = tf_v2.compat.v1.unstack(
        tf_v2.compat.v1.transpose(lstm_outputs, perm=[1, 0, 2])
    )
    lstm_outputs = tf_v2.compat.v1.concat(
        lstm_outputs,
        axis=0
    )
    mask = tf_v2.compat.v1.sequence_mask(
        lengths=self._sentence_lengths,
        maxlen=MAX_SENTENCE_LENGTH,
        dtype=tf_v2.compat.v1.float32
    ) 
    mask = tf_v2.compat.v1.concat(tf_v2.compat.v1.unstack(mask, axis=0), axis=0)
    mask = tf_v2.compat.v1.expand_dims(mask, -1)

    lstm_outputs = mask * lstm_outputs
    lstm_outputs_split = tf_v2.compat.v1.split(lstm_outputs, num_or_size_splits=self._batch_size)
    lstm_outputs_sum = tf_v2.compat.v1.reduce_sum(lstm_outputs_split, axis=1)
    lstm_outputs_average = lstm_outputs_sum / tf_v2.compat.v1.expand_dims(
        tf_v2.compat.v1.cast(self._sentence_lengths, tf_v2.compat.v1.float32),
        -1
    ) 

    return lstm_outputs_average

  def build_graph(self):
    embeddings = self.embedding_layer(self._data)
    lstm_outputs = self.LSTM_layer(embeddings)

    weigths = tf_v2.compat.v1.get_variable(
        name = 'final_layer_weights',
        shape = (self._lstm_size, NUM_CLASSES),
        initializer = tf_v2.compat.v1.random_normal_initializer(seed = 2021)
    )
    biases = tf_v2.compat.v1.get_variable(
        name = 'final_layer_biases',
        shape = (NUM_CLASSES),
        initializer = tf_v2.compat.v1.random_normal_initializer(seed = 2021)
    )
    logits = tf_v2.compat.v1.matmul(lstm_outputs, weigths) + biases

    labels_one_hot = tf_v2.compat.v1.one_hot(
        indices = self._labels,
        depth = NUM_CLASSES,
        dtype = tf_v2.compat.v1.float32
    )

    loss = tf_v2.compat.v1.nn.softmax_cross_entropy_with_logits(
        labels = labels_one_hot,
        logits = logits
    )
    loss = tf_v2.compat.v1.reduce_mean(loss)

    probs = tf_v2.compat.v1.nn.softmax(logits)
    predicted_labels = tf_v2.compat.v1.argmax(probs, axis = 1)
    predicted_labels = tf_v2.compat.v1.squeeze(predicted_labels)
    return predicted_labels, loss


  def trainer(self, loss, learning_rate):
    train_op = tf_v2.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op