import tensorflow as tf
from tensorflow.python.keras.layers import Embedding
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

import ipdb

import config


tf.enable_eager_execution()


class Policy:
    """
    Our policy. Takes as input the current state as a list of
    input/feedback indices and outputs the action distribution.
    """

    def __init__(self):
        self.guess_embedding = Embedding(config.max_guesses + 1, 
                                         config.guess_embedding_size)
        self.feedback_embedding = Embedding(config.max_feedback + 1,
                                            config.feedback_embedding_size)
        self.lstm = MultiRNNCell([
            LSTMCell(config.lstm_hidden_size), 
            LSTMCell(config.lstm_hidden_size)
        ])

        self.dense = tf.layers.Dense(config.max_guesses)

    def __call__(self, game_state):
        """
        Do a forward pass to get the action distribution
        """

        state = self.lstm.zero_state(1, tf.float32)

        for guess, feedback in game_state:
            guess_tensor = tf.reshape(tf.convert_to_tensor(guess), (1,))
            feedback_tensor = tf.reshape(tf.convert_to_tensor(feedback), (1,))
            guess_embedded = self.guess_embedding(guess_tensor)
            feedback_embedded = self.feedback_embedding(feedback_tensor)

            combined_embedded = tf.concat([guess_embedded,
                                            feedback_embedded],
                                            axis=-1)
            # can I do multiple inputs to the LSTM instead of concatenating?

            output, state = self.lstm(combined_embedded, state)

        return tf.nn.softmax(self.dense(output))


if __name__ == "__main__":
    p = Policy()
    x = p([(0, 1), (2, 3), (4, 5)])
    print(x.numpy())

