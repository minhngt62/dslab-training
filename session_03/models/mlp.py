import tensorflow as tf
# disable eager execution
tf.compat.v1.disable_eager_execution()

class MLP:
    def __init__(self, vocab_size, hidden_size, num_classes=10):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self.num_classes = num_classes

    def build_graph(self):
        self._X = tf.compat.v1.placeholder(tf.float32, shape=(None, self._vocab_size))
        self._real_Y = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        weights_1 = tf.compat.v1.get_variable(
            name="weights_input_hidden",
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.random_normal_initializer(seed=42)
            )
        biases_1 = tf.compat.v1.get_variable(
            name="biases_output_hidden",
            shape=(self._hidden_size),
            initializer=tf.random_normal_initializer(seed=42)
            )
        weights_2 = tf.compat.v1.get_variable(
            name="weights_output_hidden",
            shape=(self._hidden_size, self.num_classes),
            initializer=tf.random_normal_initializer(seed=42)
            )
        biases_2 = tf.compat.v1.get_variable(
            name="biases_input_hidden",
            shape=(self.num_classes),
            initializer=tf.random_normal_initializer(seed=42)
            )

        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2

        labels_one_hot = tf.one_hot(
            indices=self._real_Y, 
            depth=self.num_classes,
            dtype=tf.float32
            )
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits
            )
        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        pred_labels = tf.argmax(probs, axis=1)
        pred_labels = tf.squeeze(pred_labels)

        return pred_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op